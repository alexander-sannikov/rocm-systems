/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stub for gin/gin_host.h — the full implementation is in the Broadcom GIN SDK
 * (not available in this ROCm-only build environment).
 *************************************************************************/

#ifndef GIN_HOST_H_
#define GIN_HOST_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "nccl.h"
#include "net_device.h"

/* GIN type selector */
#define NCCL_GIN_TYPE_GDAKI 0
#define NCCL_GIN_TYPE_PROXY 1

/* Signal operation types for iputSignal */
#define NCCL_NET_SIGNAL_OP_INC 0
#define NCCL_NET_SIGNAL_OP_ADD 1

/* GIN net device types (extends ncclNetDeviceType enum) */
#define NCCL_NET_DEVICE_GIN_GDAKI ((ncclNetDeviceType)2)
#define NCCL_NET_DEVICE_GIN_PROXY ((ncclNetDeviceType)3)

/* GIN GDAKI device handle version */
#define NCCL_GIN_GDAKI_VERSION 1

/* GIN configuration passed to createContext */
typedef struct {
  int nSignals;
  int nCounters;
  int nContexts;
  int queueDepth;
  int trafficClass;
} ncclGinConfig_v13_t;

/* Opaque types used only in gdaki implementation (not needed by gin.cc directly) */
struct ncclGinGdakiMemHandle;
struct ncclGinGdakiGPUContext;

/* GIN collective communication plugin interface.
 * Function pointer signatures match gin.cc's actual implementations. */
typedef struct ncclGin {
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* opaqueHandle, void** listenComm);
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  ncclResult_t (*createContext)(void* collComm, ncclGinConfig_v13_t* config, void** ginCtx, ncclNetDeviceHandle_t** devHandle);
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void** ginHandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle, void** ginHandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  ncclResult_t (*destroyContext)(void* ginCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
  /* iput: ginCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request */
  ncclResult_t (*iput)(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
                       size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request);
  /* iputSignal: ginCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank,
   *             signalOff, signalMhandle, signalValue, signalOp, request */
  ncclResult_t (*iputSignal)(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
                             size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank,
                             uint64_t signalOff, void* signalMhandle, uint64_t signalValue,
                             uint32_t signalOp, void** request);
  /* iget: ginCtx, context, remoteOffset, remoteMhandle, size, localOffset, localMhandle, rank, request */
  ncclResult_t (*iget)(void* ginCtx, int context, uint64_t remoteOffset, void* remoteMhandle,
                       size_t size, uint64_t localOffset, void* localMhandle, uint32_t rank, void** request);
  /* iflush: ginCtx, context, mhandle, rank, request */
  ncclResult_t (*iflush)(void* ginCtx, int context, void* mhandle, uint32_t rank, void** request);
  /* test: collComm, request, done */
  ncclResult_t (*test)(void* collComm, void* request, int* done);
  ncclResult_t (*progress)(void* collComm);
  ncclResult_t (*queryLastError)(void* ginCtx, bool* hasError);
  ncclResult_t (*finalize)(void* ctx);
} ncclGin_t;

#endif /* GIN_HOST_H_ */
