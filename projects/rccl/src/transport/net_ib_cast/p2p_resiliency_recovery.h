/*************************************************************************
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_P2P_RESILIENCY_RECOVERY_H_
#define NET_IB_P2P_RESILIENCY_RECOVERY_H_

#include "nccl.h" // For ncclResult_t
#include "p2p_resiliency.h"

ncclResult_t IbCastPortRecoveryThreadStart();
ncclResult_t IbCastPortRecoveryThreadStop();
ncclResult_t IbCastPortRecoveryInit(struct IbCastResiliency* resCtx);
ncclResult_t IbCastPortRecoveryClose(struct IbCastResiliency* resCtx);

ncclResult_t IbCastPortRecoveryDevInit(struct IbCastResiliency* resCtx, int devIndex, IbCastDev* ibDev);
ncclResult_t IbCastPortRecoveryDevDestroy(struct IbCastResiliency* resCtx, int devIndex);

ncclResult_t IbCastPortRecoverySenderQpsCreate(struct IbCastResiliency* resCtx, struct IbCastQpInfo* localResiliencyInfo, int nQps);
ncclResult_t IbCastPortRecoverySenderQpsToRts(struct IbCastResiliency* resCtx, struct IbCastConnectionMetadata* remInfo, int nQps);

ncclResult_t IbCastPortRecoveryReceiverQpsCreateToRts(struct IbCastResiliency* resCtx, struct IbCastConnectionMetadata* remInfo, struct IbCastQpInfo* localPortRecoveryQpsInfo, int nQps);

ncclResult_t IbCastPortRecoveryQpsDestroy(struct IbCastResiliency* resCtx, int nQps);

ncclResult_t IbCastPortRecoveryHandleFailure(struct IbCastResiliency* resCtx, int devIndex);

#endif // NET_IB_P2P_RESILIENCY_RECOVERY_H_
