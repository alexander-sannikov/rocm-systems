/*************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PLUGIN_H_
#define NCCL_PLUGIN_H_

#include "nccl.h"

enum ncclPluginType {
  ncclPluginTypeNet,
  ncclPluginTypeGin,      // RCCL: added for NCCL 2.30.4 GIN host API compatibility
  ncclPluginTypeTuner,
  ncclPluginTypeProfiler,
};

void* ncclOpenNetPluginLib(const char* name);
void* ncclOpenGinPluginLib(const char* name); // RCCL: added for NCCL 2.30.4 GIN host API
void* ncclOpenTunerPluginLib(const char* name);
void* ncclOpenProfilerPluginLib(const char* name);
void* ncclGetNetPluginLib(enum ncclPluginType type);
ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type);

extern char* ncclPluginLibPaths[];

#endif
