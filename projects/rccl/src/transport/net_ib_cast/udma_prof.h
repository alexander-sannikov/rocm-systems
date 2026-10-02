/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *************************************************************************/

#ifndef RCCL_NET_IB_CAST_UDMA_PROF_H_
#define RCCL_NET_IB_CAST_UDMA_PROF_H_

#include "param.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IB_CAST_UDMA_PROF_HOST_MAX 64
#define IB_CAST_UDMA_PROF_MAX_DEVS 32

RCCL_PARAM_DECLARE(IbCastUdmaProf);

static inline int IbCastUdmaProfEnabled(void) {
  return rcclParamIbCastUdmaProf() != 0;
}

void IbCastUdmaProfLocalId(char* host, size_t hostLen, int* pid);

void IbCastUdmaProfRegisterQp(const void* commKey, int qpIndex, uint32_t qpn, int ibDevN,
                              const char* nicName, int8_t udma, int isP2p, int isDataQp,
                              int channelId, int qpIndexInDev, const char* destHost, int destPid);

void IbCastUdmaProfUpdateQpn(const void* commKey, int qpIndex, uint32_t newQpn);

void IbCastUdmaProfAddBytes(int ibDevN, int8_t udma, int isP2p, int isDataQp, uint64_t bytes);

void IbCastUdmaProfDump(void);

void IbCastUdmaProfDumpPath(char* buf, size_t cap);

void IbCastUdmaProfResetForTest(void);

size_t IbCastUdmaProfFormatJson(char* buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* RCCL_NET_IB_CAST_UDMA_PROF_H_ */
