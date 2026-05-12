/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_CORE__TYPES_H_
#define _NCCL_DEVICE_CORE__TYPES_H_
#include "../core_tmp.h"

// Per-segment window info for multi-segment VA allocations (Item J).
// ginWins uses void* (== ncclGinWindow_t) to avoid pulling host-only GIN headers
// into device-compiled units. NCCL_GIN_MAX_CONNECTIONS == 4.
struct ncclSegmentWindow {
  void* ginWins[4];
  size_t segmentSize;
};

// nccl.h has: typedef ncclWindow_vidmem* ncclWindow_t;
struct ncclWindow_vidmem {
  void* winHost;
  char* lsaFlatBase; // pointer to first byte for rank 0 of lsa team
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
  // Item J: pointer to per-segment window array (allocated in shadow pool).
  // nullptr when numGinSegments == 1 (the RCCL proxy-only common case).
  struct ncclSegmentWindow* ginMultiSegmentWins;
};

struct ncclMultimemHandle {
  void* mcBasePtr;
};

#endif
