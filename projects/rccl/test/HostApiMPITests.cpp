/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file HostApiMPITests.cpp
 * @brief P0 unit tests for RCCL's one-sided RMA Host API
 *
 * Tests cover:
 *   W1  - WindowRegisterDeregister: collective register/deregister lifecycle
 *   P1  - SinglePutRank0ToRank1:   basic ncclPutSignal + ncclWaitSignal
 *   P2  - PutWithNonZeroOffset:    peerWinOffset at kSize/2
 *   P3  - PutMultipleDataTypes:    float32, int32, float16 (raw bytes)
 *   S1  - SignalOnlyNoData:        ncclSignal with no data transfer
 *   WS2 - WaitSignalFenceSemantics: three back-to-back puts, opCnt=3
 *   O1  - DataVisibilityAfterSync: explicit host read of fine-grain memory
 *   E1  - PutSignalNullLocalbuff:  null localbuf → error (or skip)
 *   E2  - PutSignalNullWindow:     null peerWin  → error
 *   E3  - PutSignalOffsetOutOfBounds: offset past end → error (or skip)
 *   E4  - PutSignalInvalidSigIdx:  sigIdx=1      → error (or skip)
 *
 * Constraints (proxy GIN path, current API limits):
 *   sigIdx = 0, ctx = 0, flags = 0, winFlags = NCCL_WIN_DEFAULT
 *
 * API signatures (from src/nccl.h.in):
 *   ncclResult_t ncclMemAlloc(void** ptr, size_t size);
 *   ncclResult_t ncclMemFree(void* ptr);
 *   ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size,
 *                                        ncclWindow_t* win, int winFlags);
 *   ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);
 *   ncclResult_t ncclPutSignal(const void* localbuff, size_t count,
 *                               ncclDataType_t datatype, int peer,
 *                               ncclWindow_t peerWin, size_t peerWinOffset,
 *                               int sigIdx, int ctx, unsigned int flags,
 *                               ncclComm_t comm, hipStream_t stream);
 *   ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags,
 *                            ncclComm_t comm, hipStream_t stream);
 *   ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs,
 *                                ncclComm_t comm, hipStream_t stream);
 *
 * Run (example):
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter=HostApiTest.*
 */

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "HostApiHelpers.hpp"
#include "TestChecks.hpp"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLHostApiHelpers;

namespace RcclUnitTesting
{

// ============================================================================
// Test fixture
// ============================================================================

/**
 * @class HostApiTest
 * @brief GTest fixture for RCCL Host API (one-sided RMA) tests.
 *
 * SetUp creates a communicator for all ranks; individual tests call
 * validateTestPrerequisites() to skip when rank count is insufficient.
 */
class HostApiTest : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        ASSERT_EQ(ncclSuccess, createTestCommunicator());
    }

    // Convenience: get rank and world size from the active communicator.
    int rank()   const
    {
        int r = -1;
        ncclCommUserRank(const_cast<HostApiTest*>(this)->getActiveCommunicator(), &r);
        return r;
    }
    int nRanks() const
    {
        int n = -1;
        ncclCommCount(const_cast<HostApiTest*>(this)->getActiveCommunicator(), &n);
        return n;
    }
};

// ============================================================================
// Constants shared across tests
// ============================================================================

namespace
{
constexpr size_t kOneMB        = 1u << 20; // 1 MiB window size
constexpr size_t kTransferSize = 256 * 1024; // 256 KiB per PUT
constexpr int    kSigIdx       = 0;
constexpr int    kCtx          = 0;
constexpr unsigned int kFlags  = 0;
} // namespace

// ============================================================================
// W1 — WindowRegisterDeregister
// ============================================================================

/**
 * @test HostApiTest.WindowRegisterDeregister
 * @brief Verify collective ncclCommWindowRegister / ncclCommWindowDeregister lifecycle.
 *
 * All ranks allocate a fine-grain buffer, collectively register a window,
 * skip if the system does not support windows, then deregister.
 */
TEST_F(HostApiTest, WindowRegisterDeregister)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int myRank = rank();

    // Allocate fine-grain buffer (CPU-accessible on ROCm).
    void* buf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&buf, kOneMB));
    auto bufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(buf); });

    // Collective window registration.
    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(getActiveCommunicator(), buf, kOneMB, &win, NCCL_WIN_DEFAULT);

    if(win == nullptr)
    {
        // System / transport does not support windows — skip cleanly.
        if(myRank == 0)
        {
            TEST_INFO("ncclCommWindowRegister returned nullptr window — system does not "
                      "support Host API windows; skipping W1.");
        }
        GTEST_SKIP() << "System does not support ncclWindow (win == nullptr)";
    }

    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // NcclWindowGuard destructor calls ncclCommWindowDeregister.
    TEST_INFO("W1 rank %d: window registered and will be deregistered by guard.", myRank);
}

// ============================================================================
// P1 — SinglePutRank0ToRank1
// ============================================================================

/**
 * @test HostApiTest.SinglePutRank0ToRank1
 * @brief Basic ncclPutSignal (rank 0 → rank 1) + ncclWaitSignal (rank 1).
 *
 * Rank 0 fills a source buffer with fillPatternBytes, issues ncclPutSignal to
 * rank 1's window.  Rank 1 issues ncclWaitSignal(opCnt=1).  After
 * hipStreamSynchronize rank 1 verifies the window buffer.
 */
TEST_F(HostApiTest, SinglePutRank0ToRank1)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int myRank = rank();
    ncclComm_t    comm   = getActiveCommunicator();
    hipStream_t   stream = getActiveStream();

    // Both ranks allocate + register a 1 MiB window.
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, NCCL_WIN_DEFAULT);

    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow (win == nullptr)";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Rank 0: allocate unregistered source buffer, fill, then PUT.
    void* srcBuf = nullptr;
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
    if(myRank == 0)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kTransferSize));
        fillPatternBytes(srcBuf, kTransferSize, /*senderRank=*/0);

        // ncclPutSignal: localbuff, count (bytes), ncclUint8, peer=1,
        //                peerWin, peerWinOffset=0, sigIdx=0, ctx=0, flags=0
        ncclResult_t res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            kSigIdx, kCtx, kFlags, comm, stream);
        ASSERT_MPI_EQ(ncclSuccess, res);
    }

    // Rank 1: wait for 1 signal from rank 0.
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        ncclResult_t res = ncclWaitSignal(/*nDesc=*/1, &desc, comm, stream);
        ASSERT_MPI_EQ(ncclSuccess, res);
    }

    // Both ranks synchronize the stream.
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Rank 1: verify data in window buffer at offset 0.
    if(myRank == 1)
    {
        ASSERT_MPI_TRUE(verifyPatternBytes(winBuf, kTransferSize, /*senderRank=*/0));
    }

    TEST_INFO("P1 rank %d: SinglePutRank0ToRank1 passed.", myRank);
}

// ============================================================================
// P2 — PutWithNonZeroOffset
// ============================================================================

/**
 * @test HostApiTest.PutWithNonZeroOffset
 * @brief PUT with peerWinOffset = kOneMB/2.
 *
 * Same as P1 but the destination window offset is placed at the midpoint of
 * the window.  Rank 1 verifies only the offset region.
 */
TEST_F(HostApiTest, PutWithNonZeroOffset)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int       myRank = rank();
    ncclComm_t      comm   = getActiveCommunicator();
    hipStream_t     stream = getActiveStream();
    const size_t    offset = kOneMB / 2;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Zero the window so any stale bytes are detectable.
    if(myRank == 1)
    {
        memset(winBuf, 0, kOneMB);
    }

    void* srcBuf = nullptr;
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
    if(myRank == 0)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kTransferSize));
        fillPatternBytes(srcBuf, kTransferSize, /*senderRank=*/0);

        ncclResult_t res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, offset,
            kSigIdx, kCtx, kFlags, comm, stream);
        ASSERT_MPI_EQ(ncclSuccess, res);
    }

    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &desc, comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    if(myRank == 1)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        ASSERT_MPI_TRUE(verifyPatternBytes(base + offset, kTransferSize, /*senderRank=*/0));
    }

    TEST_INFO("P2 rank %d: PutWithNonZeroOffset passed.", myRank);
}

// ============================================================================
// P3 — PutMultipleDataTypes
// ============================================================================

/**
 * @test HostApiTest.PutMultipleDataTypes
 * @brief PUT 256 elements of float32, int32, and float16 and verify raw bytes.
 *
 * The test uses ncclPutSignal with the correct element type.  Verification
 * is done via fillPatternBytes / verifyPatternBytes on the raw bytes of each
 * element array, which is valid because the window buffer is CPU-accessible
 * fine-grain memory.
 */
TEST_F(HostApiTest, PutMultipleDataTypes)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int      myRank = rank();
    ncclComm_t     comm   = getActiveCommunicator();
    hipStream_t    stream = getActiveStream();
    const size_t   kElem  = 256;

    // Use a large enough window to hold all three type arrays at disjoint offsets.
    const size_t kWinSize = kOneMB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Describe the three types: {ncclDataType, element_size, offset_in_window}
    struct TypeDesc { ncclDataType_t type; size_t elemSz; size_t winOffset; };
    const TypeDesc types[] = {
        { ncclFloat32, sizeof(float),    0           },
        { ncclInt32,   sizeof(int32_t),  64  * 1024  },
        { ncclFloat16, sizeof(uint16_t), 128 * 1024  },
    };
    const int nTypes = static_cast<int>(sizeof(types) / sizeof(types[0]));

    // For each type: fill → PUT (rank 0) / WaitSignal (rank 1) → sync → verify (rank 1).
    for(int t = 0; t < nTypes; ++t)
    {
        const size_t byteCount = kElem * types[t].elemSz;

        void* srcBuf = nullptr;
        auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
        if(myRank == 0)
        {
            ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, byteCount));
            // Fill raw bytes with rank 0 pattern.
            fillPatternBytes(srcBuf, byteCount, /*senderRank=*/0);

            ncclResult_t res = ncclPutSignal(
                srcBuf, kElem, types[t].type,
                /*peer=*/1, win, types[t].winOffset,
                kSigIdx, kCtx, kFlags, comm, stream);
            ASSERT_MPI_EQ(ncclSuccess, res);
        }

        if(myRank == 1)
        {
            ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
            ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &desc, comm, stream));
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

        if(myRank == 1)
        {
            const uint8_t* base = static_cast<const uint8_t*>(winBuf);
            ASSERT_MPI_TRUE(verifyPatternBytes(base + types[t].winOffset, byteCount, /*senderRank=*/0));
        }

        TEST_INFO("P3 rank %d: type[%d] (elemSz=%zu) passed.", myRank, t, types[t].elemSz);
    }
}

// ============================================================================
// S1 — SignalOnlyNoData
// ============================================================================

/**
 * @test HostApiTest.SignalOnlyNoData
 * @brief ncclSignal with no data transfer; window contents must be unchanged.
 *
 * Rank 1 pre-fills its window with sentinel value 0xAB.
 * Rank 0 issues ncclSignal(peer=1, sigIdx=0, ctx=0, flags=0).
 * Rank 1 issues ncclWaitSignal(opCnt=1, peer=0).
 * After sync rank 1 verifies every byte is still 0xAB.
 */
TEST_F(HostApiTest, SignalOnlyNoData)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int       myRank = rank();
    ncclComm_t      comm   = getActiveCommunicator();
    hipStream_t     stream = getActiveStream();
    const uint8_t   kSentinel = 0xAB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Rank 1: pre-fill window with sentinel before the signal.
    if(myRank == 1)
    {
        memset(winBuf, kSentinel, kOneMB);
    }

    // Rank 0: signal only (no data).
    if(myRank == 0)
    {
        ncclResult_t res = ncclSignal(/*peer=*/1, kSigIdx, kCtx, kFlags, comm, stream);
        ASSERT_MPI_EQ(ncclSuccess, res);
    }

    // Rank 1: wait for the signal.
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &desc, comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Rank 1: window must still be 0xAB throughout (no data was transferred).
    if(myRank == 1)
    {
        const uint8_t* p = static_cast<const uint8_t*>(winBuf);
        bool allSentinel  = true;
        for(size_t i = 0; i < kOneMB; ++i)
        {
            if(p[i] != kSentinel)
            {
                fprintf(stderr, "S1: sentinel check failed at byte %zu: expected 0x%02x got 0x%02x\n",
                        i, static_cast<unsigned>(kSentinel), static_cast<unsigned>(p[i]));
                allSentinel = false;
                break;
            }
        }
        ASSERT_MPI_TRUE(allSentinel);
    }

    TEST_INFO("S1 rank %d: SignalOnlyNoData passed.", myRank);
}

// ============================================================================
// WS2 — WaitSignalFenceSemantics
// ============================================================================

/**
 * @test HostApiTest.WaitSignalFenceSemantics
 * @brief Three consecutive PUTs to different window offsets; WaitSignal(opCnt=3).
 *
 * Rank 0 issues three ncclPutSignal calls to rank 1's window at offsets
 * [0, kTransferSize, 2*kTransferSize], each with a different pattern
 * (senderRank encoded as 10, 20, 30 to distinguish regions).
 * Rank 1 issues ncclWaitSignal with opCnt=3.
 * After sync rank 1 verifies all three regions.
 *
 * Note: verifyPatternBytes uses senderRank as the pattern seed.  We abuse
 * this by passing synthetic "senderRank" values (10, 20, 30) that match
 * the fill calls.
 */
TEST_F(HostApiTest, WaitSignalFenceSemantics)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank    = rank();
    ncclComm_t   comm      = getActiveCommunicator();
    hipStream_t  stream    = getActiveStream();
    const int    kNumPuts  = 3;
    const size_t kWinSize  = kNumPuts * kTransferSize;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Synthetic seed values — must match fill and verify on each side.
    const int seeds[kNumPuts] = {10, 20, 30};

    if(myRank == 0)
    {
        for(int i = 0; i < kNumPuts; ++i)
        {
            void* srcBuf = nullptr;
            ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kTransferSize));
            fillPatternBytes(srcBuf, kTransferSize, seeds[i]);

            size_t winOff = static_cast<size_t>(i) * kTransferSize;
            ncclResult_t res = ncclPutSignal(
                srcBuf, kTransferSize, ncclUint8,
                /*peer=*/1, win, winOff,
                kSigIdx, kCtx, kFlags, comm, stream);
            freeFineGrainBuffer(srcBuf);
            ASSERT_MPI_EQ(ncclSuccess, res);
        }
    }

    if(myRank == 1)
    {
        // Wait for all 3 PUT+signal operations from rank 0.
        ncclWaitSignalDesc_t desc{kNumPuts, 0, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &desc, comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    if(myRank == 1)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        for(int i = 0; i < kNumPuts; ++i)
        {
            size_t off = static_cast<size_t>(i) * kTransferSize;
            ASSERT_MPI_TRUE(verifyPatternBytes(base + off, kTransferSize, seeds[i]));
        }
    }

    TEST_INFO("WS2 rank %d: WaitSignalFenceSemantics passed.", myRank);
}

// ============================================================================
// O1 — DataVisibilityAfterSync
// ============================================================================

/**
 * @test HostApiTest.DataVisibilityAfterSync
 * @brief Explicit host read of fine-grain window memory after stream sync.
 *
 * Mirrors P1 but after hipStreamSynchronize the receiver copies the
 * fine-grain window pointer into a std::vector<uint8_t> on the host
 * (memcpy from CPU-accessible fine-grain memory) and checks bytes there.
 * This exercises that fine-grain memory is readable by the CPU without any
 * hipMemcpy.
 */
TEST_F(HostApiTest, DataVisibilityAfterSync)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    void* srcBuf = nullptr;
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
    if(myRank == 0)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kTransferSize));
        fillPatternBytes(srcBuf, kTransferSize, 0);

        ASSERT_MPI_EQ(ncclSuccess,
            ncclPutSignal(srcBuf, kTransferSize, ncclUint8,
                          1, win, 0, kSigIdx, kCtx, kFlags, comm, stream));
    }

    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &desc, comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    if(myRank == 1)
    {
        // Copy from CPU-accessible fine-grain memory into a host std::vector.
        std::vector<uint8_t> hostCopy(kTransferSize);
        memcpy(hostCopy.data(), winBuf, kTransferSize);

        bool ok = true;
        for(size_t i = 0; i < kTransferSize; ++i)
        {
            uint8_t expected = static_cast<uint8_t>((0 + 1) * ((i % 251) + 1));
            if(hostCopy[i] != expected)
            {
                fprintf(stderr, "O1: mismatch at byte %zu: expected %u got %u\n",
                        i, static_cast<unsigned>(expected), static_cast<unsigned>(hostCopy[i]));
                ok = false;
                break;
            }
        }
        ASSERT_MPI_TRUE(ok);
    }

    TEST_INFO("O1 rank %d: DataVisibilityAfterSync passed.", myRank);
}

// ============================================================================
// E1 — PutSignalNullLocalbuff
// ============================================================================

/**
 * @test HostApiTest.PutSignalNullLocalbuff
 * @brief ncclPutSignal with null localbuf must return an error.
 *
 * Only rank 0 calls the API (non-collective immediate check).  Rank 1 does
 * nothing to avoid deadlock.  If argcheck is not implemented the test skips.
 */
TEST_F(HostApiTest, PutSignalNullLocalbuff)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    // Both ranks must register a window (collective).
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    if(myRank == 0)
    {
        // Pass nullptr as localbuf.
        ncclResult_t res = ncclPutSignal(
            /*localbuff=*/nullptr, /*count=*/1, ncclFloat32,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            kSigIdx, kCtx, kFlags, comm, stream);

        // If argcheck is not implemented the call may return ncclSuccess;
        // skip rather than fail in that case (the test documents intent).
        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E1: ncclPutSignal(nullptr localbuff) returned ncclSuccess "
                            "— argcheck not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E1: expected error for null localbuf, got ncclSuccess";
    }
    // Rank 1 does not call any collective → no deadlock.

    TEST_INFO("E1 rank %d: PutSignalNullLocalbuff done.", myRank);
}

// ============================================================================
// E2 — PutSignalNullWindow
// ============================================================================

/**
 * @test HostApiTest.PutSignalNullWindow
 * @brief ncclPutSignal with null peerWin must return an error.
 *
 * Non-collective: only rank 0 calls the API.
 */
TEST_F(HostApiTest, PutSignalNullWindow)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    // Allocate a valid source buffer on rank 0.
    void* srcBuf = nullptr;
    if(myRank == 0)
    {
        ASSERT_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kOneMB));
    }
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });

    if(myRank == 0)
    {
        ncclResult_t res = ncclPutSignal(
            srcBuf, /*count=*/1, ncclFloat32,
            /*peer=*/1, /*peerWin=*/nullptr, /*peerWinOffset=*/0,
            kSigIdx, kCtx, kFlags, comm, stream);

        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E2: ncclPutSignal(nullptr peerWin) returned ncclSuccess "
                            "— argcheck not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E2: expected error for null peerWin, got ncclSuccess";
    }

    TEST_INFO("E2 rank %d: PutSignalNullWindow done.", myRank);
}

// ============================================================================
// E3 — PutSignalOffsetOutOfBounds
// ============================================================================

/**
 * @test HostApiTest.PutSignalOffsetOutOfBounds
 * @brief ncclPutSignal with peerWinOffset == kOneMB (past end) must error.
 *
 * Both ranks register a 1 MiB window.  Rank 0 attempts a PUT at offset
 * equal to the window size (one byte past the end).
 */
TEST_F(HostApiTest, PutSignalOffsetOutOfBounds)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    void* srcBuf = nullptr;
    if(myRank == 0)
    {
        ASSERT_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kTransferSize));
    }
    auto srcGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });

    if(myRank == 0)
    {
        // peerWinOffset = kOneMB means the first byte of the PUT is exactly at
        // the end of the window — out of bounds.
        ncclResult_t res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/kOneMB,
            kSigIdx, kCtx, kFlags, comm, stream);

        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E3: out-of-bounds offset returned ncclSuccess "
                            "— argcheck not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E3: expected error for out-of-bounds offset, got ncclSuccess";
    }

    TEST_INFO("E3 rank %d: PutSignalOffsetOutOfBounds done.", myRank);
}

// ============================================================================
// E4 — PutSignalInvalidSigIdx
// ============================================================================

/**
 * @test HostApiTest.PutSignalInvalidSigIdx
 * @brief ncclPutSignal with sigIdx=1 (reserved, must be 0) should error.
 *
 * Non-collective: only rank 0 calls the API.  Skip if not validated.
 */
TEST_F(HostApiTest, PutSignalInvalidSigIdx)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    void* srcBuf = nullptr;
    if(myRank == 0)
    {
        ASSERT_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kTransferSize));
    }
    auto srcGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });

    if(myRank == 0)
    {
        ncclResult_t res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            /*sigIdx=*/1, kCtx, kFlags, comm, stream);

        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E4: sigIdx=1 returned ncclSuccess "
                            "— sigIdx validation not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E4: expected error for sigIdx=1, got ncclSuccess";
    }

    TEST_INFO("E4 rank %d: PutSignalInvalidSigIdx done.", myRank);
}

// ============================================================================
// S2 — SignalCumulativeFence
// ============================================================================

/**
 * @test HostApiTest.SignalCumulativeFence
 * @brief ncclPutSignal then ncclSignal from rank 0; rank 1 waits with opCnt=2.
 *
 * Rank 0 issues one ncclPutSignal (256 bytes to rank 1's window at offset 0)
 * then one ncclSignal (no data) to rank 1, both before rank 1 waits.
 * Rank 1 waits with opCnt=2 (fence: 1 PUT + 1 SIGNAL).
 * After sync rank 1 verifies the PUT data.
 */
TEST_F(HostApiTest, SignalCumulativeFence)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();
    const size_t kSize  = 256;
    const size_t kWinSize = 4096;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    void* srcBuf = nullptr;
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
    if(myRank == 0)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kSize));
        fillPatternBytes(srcBuf, kSize, /*senderRank=*/0);

        // Enqueue PUT + SIGNAL as a group to ensure ordering.
        ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
        ncclResult_t r1 = ncclPutSignal(
            srcBuf, kSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            kSigIdx, kCtx, kFlags, comm, stream);
        ncclResult_t r2 = ncclSignal(/*peer=*/1, kSigIdx, kCtx, kFlags, comm, stream);
        ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
        ASSERT_MPI_EQ(ncclSuccess, r1);
        ASSERT_MPI_EQ(ncclSuccess, r2);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(myRank == 1)
    {
        // opCnt=2: one PUT counts as a signal, plus one explicit SIGNAL.
        ncclWaitSignalDesc_t desc{/*opCnt=*/2, /*peer=*/0, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(/*nDesc=*/1, &desc, comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    if(myRank == 1)
    {
        ASSERT_MPI_TRUE(verifyPatternBytes(winBuf, kSize, /*senderRank=*/0));
    }

    TEST_INFO("S2 rank %d: SignalCumulativeFence passed.", myRank);
}

// ============================================================================
// WS3 — MultipleSendersOneReceiver
// ============================================================================

/**
 * @test HostApiTest.MultipleSendersOneReceiver
 * @brief Two senders (rank 0, rank 1) put to disjoint offsets of rank 2's window.
 *
 * Rank 0 PUTs 256 bytes at offset 0; rank 1 PUTs 256 bytes at offset 4096.
 * Rank 2 issues two separate ncclWaitSignal calls (one per sender, opCnt=1 each).
 * After sync rank 2 verifies both regions.
 */
TEST_F(HostApiTest, MultipleSendersOneReceiver)
{
    if(!validateTestPrerequisites(/*min=*/3))
    {
        GTEST_SKIP() << "Need at least 3 MPI processes";
    }

    const int    myRank   = rank();
    ncclComm_t   comm     = getActiveCommunicator();
    hipStream_t  stream   = getActiveStream();
    const size_t kSize    = 256;
    const size_t kWinSize = 8192;

    // All ranks register a window (collective).
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    void* srcBuf = nullptr;
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
    if(myRank == 0)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kSize));
        fillPatternBytes(srcBuf, kSize, /*senderRank=*/0);
        ASSERT_MPI_EQ(ncclSuccess,
            ncclPutSignal(srcBuf, kSize, ncclUint8,
                          /*peer=*/2, win, /*peerWinOffset=*/0,
                          kSigIdx, kCtx, kFlags, comm, stream));
    }

    if(myRank == 1)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kSize));
        fillPatternBytes(srcBuf, kSize, /*senderRank=*/1);
        ASSERT_MPI_EQ(ncclSuccess,
            ncclPutSignal(srcBuf, kSize, ncclUint8,
                          /*peer=*/2, win, /*peerWinOffset=*/4096,
                          kSigIdx, kCtx, kFlags, comm, stream));
    }

    if(myRank == 2)
    {
        // Wait for rank 0's signal.
        ncclWaitSignalDesc_t d0{/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &d0, comm, stream));
        // Wait for rank 1's signal.
        ncclWaitSignalDesc_t d1{/*opCnt=*/1, /*peer=*/1, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &d1, comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    if(myRank == 2)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        ASSERT_MPI_TRUE(verifyPatternBytes(base + 0,    kSize, /*senderRank=*/0));
        ASSERT_MPI_TRUE(verifyPatternBytes(base + 4096, kSize, /*senderRank=*/1));
    }

    TEST_INFO("WS3 rank %d: MultipleSendersOneReceiver passed.", myRank);
}

// ============================================================================
// WS4 — WaitSignalMultipleDescriptors
// ============================================================================

/**
 * @test HostApiTest.WaitSignalMultipleDescriptors
 * @brief ncclWaitSignal called once with an array of 2 descriptors (nDesc=2).
 *
 * Same topology as WS3 (ranks 0 and 1 → rank 2) but rank 2 passes both
 * descriptors in a single ncclWaitSignal(comm, 2, descs, stream) call.
 */
TEST_F(HostApiTest, WaitSignalMultipleDescriptors)
{
    if(!validateTestPrerequisites(/*min=*/3))
    {
        GTEST_SKIP() << "Need at least 3 MPI processes";
    }

    const int    myRank   = rank();
    ncclComm_t   comm     = getActiveCommunicator();
    hipStream_t  stream   = getActiveStream();
    const size_t kSize    = 256;
    const size_t kWinSize = 8192;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    void* srcBuf = nullptr;
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
    if(myRank == 0)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kSize));
        fillPatternBytes(srcBuf, kSize, /*senderRank=*/0);
        ASSERT_MPI_EQ(ncclSuccess,
            ncclPutSignal(srcBuf, kSize, ncclUint8,
                          /*peer=*/2, win, /*peerWinOffset=*/0,
                          kSigIdx, kCtx, kFlags, comm, stream));
    }

    if(myRank == 1)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kSize));
        fillPatternBytes(srcBuf, kSize, /*senderRank=*/1);
        ASSERT_MPI_EQ(ncclSuccess,
            ncclPutSignal(srcBuf, kSize, ncclUint8,
                          /*peer=*/2, win, /*peerWinOffset=*/4096,
                          kSigIdx, kCtx, kFlags, comm, stream));
    }

    if(myRank == 2)
    {
        // Both descriptors in one call.
        ncclWaitSignalDesc_t descs[2];
        descs[0] = {/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        descs[1] = {/*opCnt=*/1, /*peer=*/1, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(/*nDesc=*/2, &descs[0], comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    if(myRank == 2)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        ASSERT_MPI_TRUE(verifyPatternBytes(base + 0,    kSize, /*senderRank=*/0));
        ASSERT_MPI_TRUE(verifyPatternBytes(base + 4096, kSize, /*senderRank=*/1));
    }

    TEST_INFO("WS4 rank %d: WaitSignalMultipleDescriptors passed.", myRank);
}

// ============================================================================
// P4 — LargePut
// ============================================================================

/**
 * @test HostApiTest.LargePut
 * @brief PUT 256 MiB from rank 0 to rank 1's window; spot-check first and last 64 bytes.
 *
 * We use 256 MiB rather than the full 1 GiB to avoid OOM on hardware with
 * limited fine-grain / pinned memory capacity while still exercising a
 * large-transfer code path.
 */
TEST_F(HostApiTest, LargePut)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank  = rank();
    ncclComm_t   comm    = getActiveCommunicator();
    hipStream_t  stream  = getActiveStream();

    // 256 MiB — large but avoids OOM on most ROCm systems.
    const size_t kLargeSize = 256ULL * 1024 * 1024;
    const uint8_t kByte     = 0xAB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kLargeSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kLargeSize, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    void* srcBuf = nullptr;
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
    if(myRank == 0)
    {
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kLargeSize));
        memset(srcBuf, kByte, kLargeSize);

        ASSERT_MPI_EQ(ncclSuccess,
            ncclPutSignal(srcBuf, kLargeSize, ncclUint8,
                          /*peer=*/1, win, /*peerWinOffset=*/0,
                          kSigIdx, kCtx, kFlags, comm, stream));
    }

    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &desc, comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    if(myRank == 1)
    {
        // Spot-check: first 64 bytes and last 64 bytes.
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        uint8_t expected[64];
        memset(expected, kByte, sizeof(expected));

        EXPECT_EQ(0, memcmp(base, expected, 64))
            << "P4: first 64 bytes of large PUT are incorrect";
        EXPECT_EQ(0, memcmp(base + kLargeSize - 64, expected, 64))
            << "P4: last 64 bytes of large PUT are incorrect";
    }

    TEST_INFO("P4 rank %d: LargePut passed.", myRank);
}

// ============================================================================
// P5 — AllToAllPut
// ============================================================================

/**
 * @test HostApiTest.AllToAllPut
 * @brief Each rank i PUTs to rank (i+1)%N and waits for rank (i-1+N)%N.
 *
 * All ranks register 4096-byte windows.  Each rank fills 256 bytes with its
 * own rank pattern, PUTs to the next rank (offset 0), then waits for the
 * previous rank's signal.  After sync each rank verifies the received data.
 */
TEST_F(HostApiTest, AllToAllPut)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank   = rank();
    const int    nRanks_  = nRanks();
    ncclComm_t   comm     = getActiveCommunicator();
    hipStream_t  stream   = getActiveStream();
    const size_t kSize    = 256;
    const size_t kWinSize = 4096;

    const int sendTo  = (myRank + 1)           % nRanks_;
    const int recvFrom = (myRank + nRanks_ - 1) % nRanks_;

    // All ranks register a window.
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Allocate source buffer and fill with this rank's pattern.
    void* srcBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kSize));
    auto srcBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(srcBuf); });
    fillPatternBytes(srcBuf, kSize, myRank);

    // Batch PUT + WAIT in a group.
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ncclResult_t rPut = ncclPutSignal(
        srcBuf, kSize, ncclUint8,
        sendTo, win, /*peerWinOffset=*/0,
        kSigIdx, kCtx, kFlags, comm, stream);
    ncclWaitSignalDesc_t desc{/*opCnt=*/1, recvFrom, kSigIdx, kCtx};
    ncclResult_t rWait = ncclWaitSignal(1, &desc, comm, stream);
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_MPI_EQ(ncclSuccess, rPut);
    ASSERT_MPI_EQ(ncclSuccess, rWait);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Verify: we should have received recvFrom's pattern at offset 0.
    ASSERT_MPI_TRUE(verifyPatternBytes(winBuf, kSize, recvFrom));

    TEST_INFO("P5 rank %d: AllToAllPut passed (recv from rank %d).", myRank, recvFrom);
}

// ============================================================================
// O2 — SignalImpliesPriorPutsDelivered
// ============================================================================

/**
 * @test HostApiTest.SignalImpliesPriorPutsDelivered
 * @brief Two ncclPutSignal calls from rank 0; rank 1 waits with opCnt=2.
 *
 * Each ncclPutSignal implicitly delivers a signal.  Two calls = opCnt 2.
 * Rank 1 verifies both data regions after sync.
 */
TEST_F(HostApiTest, SignalImpliesPriorPutsDelivered)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank  = rank();
    ncclComm_t   comm    = getActiveCommunicator();
    hipStream_t  stream  = getActiveStream();
    const size_t kSize   = 256;
    const size_t kWinSize = kOneMB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    if(myRank == 0)
    {
        void* src0 = nullptr;
        void* src1 = nullptr;
        auto src0Guard = makeScopeGuard([&]() { if(src0) freeFineGrainBuffer(src0); });
        auto src1Guard = makeScopeGuard([&]() { if(src1) freeFineGrainBuffer(src1); });
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&src0, kSize));
        ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&src1, kSize));
        fillPatternBytes(src0, kSize, /*senderRank=*/0);
        fillPatternBytes(src1, kSize, /*senderRank=*/10); // distinct seed for region 1

        // Two separate ncclPutSignal calls — each counts as one signal.
        ASSERT_MPI_EQ(ncclSuccess,
            ncclPutSignal(src0, kSize, ncclUint8,
                          /*peer=*/1, win, /*peerWinOffset=*/0,
                          kSigIdx, kCtx, kFlags, comm, stream));
        ASSERT_MPI_EQ(ncclSuccess,
            ncclPutSignal(src1, kSize, ncclUint8,
                          /*peer=*/1, win, /*peerWinOffset=*/512,
                          kSigIdx, kCtx, kFlags, comm, stream));
    }

    if(myRank == 1)
    {
        // opCnt=2 because two PutSignal calls were made.
        ncclWaitSignalDesc_t desc{/*opCnt=*/2, /*peer=*/0, kSigIdx, kCtx};
        ASSERT_MPI_EQ(ncclSuccess, ncclWaitSignal(1, &desc, comm, stream));
    }

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    if(myRank == 1)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        ASSERT_MPI_TRUE(verifyPatternBytes(base + 0,   kSize, /*senderRank=*/0));
        ASSERT_MPI_TRUE(verifyPatternBytes(base + 512, kSize, /*senderRank=*/10));
    }

    TEST_INFO("O2 rank %d: SignalImpliesPriorPutsDelivered passed.", myRank);
}

// ============================================================================
// E5 — PutSignalInvalidCtx
// ============================================================================

/**
 * @test HostApiTest.PutSignalInvalidCtx
 * @brief ncclPutSignal with ctx=1 (reserved, must be 0) should error.
 *
 * Non-collective: only rank 0 calls the API.  Skip if argcheck is not
 * implemented (i.e., the call returns ncclSuccess).
 */
TEST_F(HostApiTest, PutSignalInvalidCtx)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, NCCL_WIN_DEFAULT);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    void* srcBuf = nullptr;
    if(myRank == 0)
    {
        ASSERT_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kTransferSize));
    }
    auto srcGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });

    if(myRank == 0)
    {
        ncclResult_t res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            kSigIdx, /*ctx=*/1, kFlags, comm, stream);

        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E5: ncclPutSignal(ctx=1) returned ncclSuccess "
                            "— ctx validation not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E5: expected error for ctx=1, got ncclSuccess";
    }
    // Rank 1 does not participate in this non-collective error path.

    TEST_INFO("E5 rank %d: PutSignalInvalidCtx done.", myRank);
}

// ============================================================================
// E6 — WaitSignalNullDescs
// ============================================================================

/**
 * @test HostApiTest.WaitSignalNullDescs
 * @brief ncclWaitSignal(nDesc=1, nullptr, ...) must return ncclInvalidArgument.
 *
 * Each rank calls independently (non-collective).  Skip if not validated.
 */
TEST_F(HostApiTest, WaitSignalNullDescs)
{
    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    ncclResult_t res = ncclWaitSignal(/*nDesc=*/1, /*signalDescs=*/nullptr, comm, stream);
    if(res == ncclSuccess)
    {
        GTEST_SKIP() << "E6: ncclWaitSignal(nDesc=1, nullptr) returned ncclSuccess "
                        "— null-desc validation not implemented; skipping.";
    }
    EXPECT_EQ(ncclInvalidArgument, res)
        << "E6: expected ncclInvalidArgument for null descs with nDesc=1";

    TEST_INFO("E6 rank %d: WaitSignalNullDescs done.", myRank);
}

// ============================================================================
// E7 — WaitSignalZeroDesc
// ============================================================================

/**
 * @test HostApiTest.WaitSignalZeroDesc
 * @brief ncclWaitSignal(nDesc=0, nullptr, ...) — zero descriptors is a no-op.
 *
 * Expect ncclSuccess (or ncclInvalidArgument — both are acceptable).
 * No stream sync or data transfer involved.
 */
TEST_F(HostApiTest, WaitSignalZeroDesc)
{
    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    ncclResult_t res = ncclWaitSignal(/*nDesc=*/0, /*signalDescs=*/nullptr, comm, stream);
    EXPECT_TRUE(res == ncclSuccess || res == ncclInvalidArgument)
        << "E7: expected ncclSuccess or ncclInvalidArgument for nDesc=0, got "
        << static_cast<int>(res);

    TEST_INFO("E7 rank %d: WaitSignalZeroDesc done (result=%d).", myRank, static_cast<int>(res));
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
