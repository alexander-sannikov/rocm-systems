// Stub implementations: gin_host_gdaki.cc requires DOCA SDK and CUDA, unavailable in ROCm-only builds.
// All entry points return ncclInternalError so callers fail gracefully at runtime.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "nccl.h"

// Forward-declare the handle type used in CreateContext — defined in nccl_net.h via nccl.h.
struct ncclNetDeviceHandle_v7_t;
typedef struct ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_t;

ncclResult_t ncclGinGdakiCreateContext(void* /*collComm*/, int /*nSignals*/, int /*nCounters*/,
                                       int /*nContexts*/, int /*queueDepth*/, int /*trafficClass*/,
                                       void** /*outGinCtx*/, ncclNetDeviceHandle_t** /*outDevHandle*/) {
  return ncclInternalError;
}

ncclResult_t ncclGinGdakiDestroyContext(void* /*ginCtx*/) {
  return ncclInternalError;
}

ncclResult_t ncclGinGdakiRegMrSym(void* /*collComm*/, void* /*data*/, size_t /*size*/, int /*type*/,
                                   uint64_t /*mr_flags*/, void** /*mhandle*/, void** /*ginHandle*/) {
  return ncclInternalError;
}

ncclResult_t ncclGinGdakiDeregMrSym(void* /*collComm*/, void* /*mhandle*/) {
  return ncclInternalError;
}

ncclResult_t ncclGinGdakiProgress(void* /*ginCtx*/) {
  return ncclInternalError;
}

ncclResult_t ncclGinGdakiQueryLastError(void* /*ginCtx*/, bool* /*hasError*/) {
  return ncclInternalError;
}
