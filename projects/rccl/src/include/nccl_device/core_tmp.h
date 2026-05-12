/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * RCCL merge shim: core_tmp.h
 *
 * NCCL's gin_device_host_common.h includes nccl_device/core.h to get
 * ncclGinSignal_t and ncclGinCounter_t.  RCCL's nccl_device/core.h is a
 * trimmed device-only header that intentionally omits those host-visible
 * GIN helper types.  This file re-exports core.h and adds the missing
 * typedefs so that every header that would normally #include "core.h" for
 * GIN signal/counter types can instead #include "core_tmp.h".
 *
 * Do NOT include CUDA/HIP runtime headers here – this header is pulled in
 * from plain host C++ translation units.
 */

#ifndef _NCCL_DEVICE_CORE_TMP_H_
#define _NCCL_DEVICE_CORE_TMP_H_

#include "nccl_device/core.h"

#include <stdint.h>

/* GIN helper types present in NCCL's core.h but absent from RCCL's trimmed
 * device/core.h.  Defined here to satisfy headers that reference them. */

/* NCCL_GIN_MAX_CONNECTIONS — also defined in gin_host_win_stub.h; mirrored
 * here so that device-side headers (comm__types.h) can use it without pulling
 * in the full host-side gin header. */
#ifndef NCCL_GIN_MAX_CONNECTIONS
#define NCCL_GIN_MAX_CONNECTIONS 4
#endif

typedef uint32_t ncclGinSignal_t;
typedef uint32_t ncclGinCounter_t;

typedef struct alignas(uint64_t) {
  char opaque[16];
} ncclGinRequest_t;

struct ncclGinBarrierHandle;
typedef struct ncclGinBarrierHandle ncclGinBarrierHandle_t;

#endif /* _NCCL_DEVICE_CORE_TMP_H_ */
