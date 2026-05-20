/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_COMM__TYPES_H_
#define _NCCL_DEVICE_COMM__TYPES_H_
#include "../comm_tmp.h"
#include "core__types.h"
#include "ll_a2a__types.h"
#include "lsa_barrier__types.h"
#include "gin_barrier__types.h"

struct ncclDevCommWindowTable;
#if __cplusplus
struct ncclDevCommWindowTable {
  struct Entry {
    uintptr_t base, size;
    ncclWindow_t window;
  } entries[32];
  struct ncclDevCommWindowTable* next;
};
#endif

struct ncclDevComm {
  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  struct ncclDevCommWindowTable* windowTable;

  ncclWindow_t resourceWindow;
  struct ncclWindow_vidmem resourceWindow_inlined;

  ncclMultimemHandle_t lsaMultimem;
  ncclLsaBarrierHandle_t lsaBarrier;
  ncclGinBarrierHandle_t railGinBarrier;

#ifndef NCCL_GIN_MAX_CONNECTIONS
#define NCCL_GIN_MAX_CONNECTIONS 4
#endif
  uint8_t ginConnectionCount;
  uint8_t ginNetDeviceTypes[NCCL_GIN_MAX_CONNECTIONS];
  void* ginHandles[NCCL_GIN_MAX_CONNECTIONS];
  int ginSignalCount;
  int ginCounterCount;
  uint32_t ginContextCount;
};

#endif // _NCCL_DEVICE_COMM__TYPES_H_
