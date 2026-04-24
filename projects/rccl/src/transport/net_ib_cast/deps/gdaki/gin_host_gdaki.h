// Stub header: gin_host_gdaki.h requires DOCA SDK and CUDA, unavailable in ROCm-only builds.
// Provides minimal declarations so gin.cc compiles; GDAKI functionality is not available.

#ifndef _GIN_HOST_GDAKI_H_
#define _GIN_HOST_GDAKI_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "nccl.h"
#include "gin/gin_host.h"

ncclResult_t ncclGinGdakiCreateContext(void *collComm, int nSignals, int nCounters, int nContexts, int queueDepth,
                                       int trafficClass, void **outGinCtx, ncclNetDeviceHandle_t **outDevHandle);
ncclResult_t ncclGinGdakiDestroyContext(void *ginCtx);
ncclResult_t ncclGinGdakiRegMrSym(void *collComm, void *data, size_t size, int type, uint64_t mr_flags, void **mhandle,
                                  void **ginHandle);
ncclResult_t ncclGinGdakiDeregMrSym(void *collComm, void *mhandle);
ncclResult_t ncclGinGdakiProgress(void *ginCtx);
ncclResult_t ncclGinGdakiQueryLastError(void *ginCtx, bool *hasError);

#endif
