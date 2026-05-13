/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file HostApiHelpers.hpp
 * @brief Helper utilities for testing RCCL's one-sided RMA (Host API)
 *
 * Provides:
 * - allocFineGrainBuffer / freeFineGrainBuffer via ncclMemAlloc / ncclMemFree
 * - NcclWindowGuard: RAII wrapper for ncclCommWindowRegister / ncclCommWindowDeregister
 * - fillPatternBytes / verifyPatternBytes for byte-level test patterns on CPU-accessible
 *   (fine-grain) memory
 *
 * Fine-grain memory allocated via ncclMemAlloc is CPU-accessible on ROCm, so
 * fillPattern / verifyPattern operate directly on the pointer without hipMemcpy.
 */

#ifndef HOST_API_HELPERS_HPP
#define HOST_API_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include "rccl/rccl.h"
#include <cstdint>
#include <cstdio>

namespace RCCLHostApiHelpers
{

// ============================================================================
// Fine-grain buffer allocation
// ============================================================================

/**
 * @brief Allocate fine-grain (CPU-accessible) memory via ncclMemAlloc.
 *
 * On ROCm the returned pointer is host-accessible without hipMemcpy.
 *
 * @param ptr   Out: receives allocated pointer.
 * @param size  Requested size in bytes.
 * @return ncclResult_t from ncclMemAlloc.
 */
inline ncclResult_t allocFineGrainBuffer(void** ptr, size_t size)
{
    return ncclMemAlloc(ptr, size);
}

/**
 * @brief Free fine-grain memory previously allocated with allocFineGrainBuffer.
 *
 * @param ptr Pointer to free (no-op if nullptr).
 * @return ncclResult_t from ncclMemFree, or ncclSuccess if ptr was nullptr.
 */
inline ncclResult_t freeFineGrainBuffer(void* ptr)
{
    if(!ptr)
        return ncclSuccess;
    return ncclMemFree(ptr);
}

// ============================================================================
// RAII guard for ncclWindow_t
// ============================================================================

/**
 * @struct NcclWindowGuard
 * @brief RAII wrapper for ncclCommWindowRegister / ncclCommWindowDeregister.
 *
 * Constructor calls ncclCommWindowRegister. Destructor calls
 * ncclCommWindowDeregister only if win_ is non-null.
 *
 * Usage:
 * @code
 *   void* buf = nullptr;
 *   allocFineGrainBuffer(&buf, kSize);
 *   ncclWindow_t win = nullptr;
 *   NcclWindowGuard wg(comm, buf, kSize, &win, NCCL_WIN_DEFAULT);
 *   if (win == nullptr) { GTEST_SKIP() << "Windows not supported"; }
 *   // ... use win ...
 *   // Destructor calls ncclCommWindowDeregister automatically.
 * @endcode
 */
struct NcclWindowGuard
{
    ncclComm_t   comm_  = nullptr;
    ncclWindow_t win_   = nullptr;
    ncclResult_t initResult_ = ncclSuccess;

    NcclWindowGuard() = default;

    /**
     * @brief Register a memory window.
     *
     * @param comm     RCCL communicator (collective across all ranks).
     * @param buff     Buffer to register (fine-grain, CPU-accessible).
     * @param size     Size of the buffer in bytes.
     * @param winOut   Out: receives the window handle.
     * @param winFlags Window flags (use NCCL_WIN_DEFAULT = 0x00 for proxy GIN path).
     */
    NcclWindowGuard(ncclComm_t    comm,
                    void*         buff,
                    size_t        size,
                    ncclWindow_t* winOut,
                    int           winFlags)
        : comm_(comm)
    {
        initResult_ = ncclCommWindowRegister(comm_, buff, size, &win_, winFlags);
        if(winOut)
            *winOut = win_;
    }

    ~NcclWindowGuard()
    {
        if(win_ && comm_)
        {
            ncclResult_t res = ncclCommWindowDeregister(comm_, win_);
            if(res != ncclSuccess)
            {
                fprintf(stderr,
                        "WARNING: ncclCommWindowDeregister failed in destructor: %s\n",
                        ncclGetErrorString(res));
            }
            win_ = nullptr;
        }
    }

    // Non-copyable, non-movable (owns the handle)
    NcclWindowGuard(const NcclWindowGuard&)            = delete;
    NcclWindowGuard& operator=(const NcclWindowGuard&) = delete;
    NcclWindowGuard(NcclWindowGuard&&)                 = delete;
    NcclWindowGuard& operator=(NcclWindowGuard&&)      = delete;

    ncclResult_t initResult() const { return initResult_; }
    ncclWindow_t win()        const { return win_; }
    bool         valid()      const { return win_ != nullptr; }
};

// ============================================================================
// Pattern fill / verify for fine-grain (CPU-accessible) memory
// ============================================================================

/**
 * @brief Fill a host-accessible (fine-grain) buffer with a rank-indexed byte pattern.
 *
 * Pattern per byte i: uint8_t((senderRank + 1) * ((i % 251) + 1))
 *
 * The modulo 251 keeps values within the range that round-trips through uint8_t
 * without collision for distinct ranks.
 *
 * @param hostBuf    CPU-accessible pointer (e.g., ncclMemAlloc fine-grain buffer).
 * @param bytes      Number of bytes to fill.
 * @param senderRank MPI rank of the sender (used as part of the pattern).
 */
inline void fillPatternBytes(void* hostBuf, size_t bytes, int senderRank)
{
    auto* p = static_cast<uint8_t*>(hostBuf);
    for(size_t i = 0; i < bytes; ++i)
    {
        p[i] = static_cast<uint8_t>((senderRank + 1) * ((i % 251) + 1));
    }
}

/**
 * @brief Verify a host-accessible (fine-grain) buffer against the rank-indexed byte pattern.
 *
 * Uses the same formula as fillPatternBytes.
 *
 * @param hostBuf    CPU-accessible pointer (e.g., ncclMemAlloc fine-grain buffer).
 * @param bytes      Number of bytes to verify.
 * @param senderRank MPI rank of the original sender (must match fill call).
 * @return true if every byte matches, false on the first mismatch.
 */
inline bool verifyPatternBytes(const void* hostBuf, size_t bytes, int senderRank)
{
    const auto* p = static_cast<const uint8_t*>(hostBuf);
    for(size_t i = 0; i < bytes; ++i)
    {
        uint8_t expected = static_cast<uint8_t>((senderRank + 1) * ((i % 251) + 1));
        if(p[i] != expected)
        {
            fprintf(stderr,
                    "verifyPatternBytes: mismatch at byte %zu: expected %u got %u "
                    "(senderRank=%d)\n",
                    i,
                    static_cast<unsigned>(expected),
                    static_cast<unsigned>(p[i]),
                    senderRank);
            return false;
        }
    }
    return true;
}

} // namespace RCCLHostApiHelpers

#endif // MPI_TESTS_ENABLED

#endif // HOST_API_HELPERS_HPP
