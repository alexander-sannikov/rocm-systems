/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "p2p_resiliency.h"

char ncclIbIfName[MAX_IF_NAME_SIZE+1];
union ncclSocketAddress ncclIbIfAddr;

int ncclNMergedIbDevs = -1;
int ncclNIbDevs = -1;
struct ncclIbMergedDev IbCastMergedDevs[MAX_IB_VDEVS];
struct ncclIbDev IbCastDevs[MAX_IB_DEVS];
int ncclIbRelaxedOrderingEnabled = 0;

bool rcclAinicRoce = 0;
bool rcclCtsInlineData = 0;
bool rcclCtsOffloadEnabled = 0;
bool ncclIbUseInline = 0;
int ncclIbGdrFlushDisable = 0;

ncclChannelToUd nccl_channel_ud_map[MAX_IB_DEVS][MAXCHANNELS][ncclIbChannelTypeMax];
bool nccl_channel_last_ud[MAX_IB_DEVS][ncclIbChannelTypeMax];

ncclProfilerCallback_t ncclProfilerFunction;

NCCL_PARAM(IbCastSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);
NCCL_PARAM(IbPrepostReceiveWorkRequests, "IB_PREPOST_RECEIVE_WORK_REQUESTS", -2);
NCCL_PARAM(IbCastAsyncEvents,"IB_RETURN_ASYNC_EVENTS",1);
extern int ncclParamIbReceiverSideMatchingScheme();
extern int ncclParamIbOooRq();
extern int ncclParamIbResiliencyPortFailover();


ncclResult_t IbCastStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbCastAsyncEvents() && __atomic_load_n(&stat->fatalErrorCount, __ATOMIC_RELAXED)) {
    ERROR("RCCL encountered a communication fatal error (detected in %s)\n", funcName);
    ERROR("RCCL cannot recover from this network failure and now exiting. Please check the network health.");
    return ncclSystemError;
  }
  return ncclSuccess;
}

struct ncclIbNetCommDevBase* IbCastGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*) base;
    return &sComm->devs[devIndex].base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*) base;
    return &rComm->devs[devIndex].base;
  }
}

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend) {
  for (int i = 0; i < NCCL_IB_MAX_QPS; i++) {
    baseComm->qps[i].devIndex= -1;
    baseComm->qps[i].remDevIdx= -1;
    baseComm->activeQps[i] = &baseComm->qps[i];
    baseComm->qps[i].eceSupported = 0;
    baseComm->qps[i].ece = {0};
    memset(&baseComm->qps[i].initAttr, 0, sizeof(baseComm->qps[i].initAttr));
    memset(&baseComm->qps[i].rtrAttr, 0, sizeof(baseComm->qps[i].rtrAttr));
    memset(&baseComm->qps[i].rtsAttr, 0, sizeof(baseComm->qps[i].rtsAttr));
  }
  baseComm->nqps = -1;
  baseComm->splitDataOnQps = ncclParamIbCastSplitDataOnQps();
  baseComm->nDataQps = -1;
  baseComm->isSend = isSend;
  baseComm->ready = 0;

  NCCLCHECK(ncclIbResiliencyInit(baseComm, &baseComm->resiliency));
  baseComm->recvMatchingScheme = ncclParamIbReceiverSideMatchingScheme() == -2 ? BY_INDEX : ncclParamIbReceiverSideMatchingScheme();

  if (ncclParamIbOooRq() || (ncclParamIbResiliencyPortFailover() == 1)) {
    baseComm->recvMatchingScheme = BY_ID;
    if (ncclParamIbReceiverSideMatchingScheme() == BY_INDEX) {
      INFO(NCCL_NET, "NET/IB: %s: Overriding matching scheme to ID-based (%d)", __func__, BY_ID);
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm) {
  NCCLCHECK(ncclIbBaseCommInit(&recvComm->base, false));
  recvComm->ibRecvWorkRequest = {
    .wr_id = NCCL_IB_RECV_WR_ID_DUMMY,
    .next = NULL,
    .sg_list = NULL,
    .num_sge = 0
  };

  recvComm->prepostReceiveWorkRequests = (ncclParamIbPrepostReceiveWorkRequests() == -2) ? false : ncclParamIbPrepostReceiveWorkRequests();

  if (recvComm->base.resiliency) {
    if (ncclParamIbPrepostReceiveWorkRequests() == 0) {
      INFO(NCCL_NET, "NET/IB: %s: Overriding pre-posting to true (1).", __func__);
    }
    recvComm->prepostReceiveWorkRequests = true;
  }
  if (ncclParamIbOooRq()) {
    if (ncclParamIbPrepostReceiveWorkRequests() == 0) {
      INFO(NCCL_NET, "NET/IB: %s: OOO RQ is enabled, Overriding pre-posting to true (1).", __func__);
    }
    recvComm->prepostReceiveWorkRequests = true;
  }

  INFO(NCCL_NET, "NET/IB: %s: Receive work requests will be %s", __func__, recvComm->prepostReceiveWorkRequests ? "pre-posted" : "posted on-demand");
  return ncclSuccess;
}

ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm) {
  NCCLCHECK(ncclIbBaseCommInit(&sendComm->base, true));
  return ncclSuccess;
}

pthread_t IbCastAsyncThread;
static void* IbCastAsyncThreadMain(void* args) {
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      ncclIbDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      // the above is a CQ fatal error
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      ncclIbCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // the above are QP fatal errors
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      ncclIbQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in NCCL
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_GID_CHANGE:
      WARN("NET/IB : %s:%d GID table changed", dev->devName, dev->portNum);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got non-fatal async event: %s(%d)", dev->devName, dev->portNum, str, event.event_type);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

ncclResult_t rcclCastNetP2pPolicy(void* handle, int isP2p) {
  if (!handle) return ncclInvalidArgument;
  struct ncclIbHandle* ibHandle = (struct ncclIbHandle*)handle;
  if (ibHandle->magic != NCCL_SOCKET_MAGIC) return ncclInvalidArgument;
  ibHandle->isP2p = isP2p;
  return ncclSuccess;
}

ncclNet_t netIbCast = {
  "IB-CAST",
  IbCastInit,
  IbCastDevices,
  IbCastGetProperties,
  IbCastListen,
  IbCastConnect,
  IbCastAccept,
  IbCastRegMr,
  IbCastRegMrDmaBuf,
  IbCastDeregMr,
  IbCastIsend,
  IbCastIrecv,
  IbCastIflush,
  IbCastTest,
  IbCastCloseSend,
  IbCastCloseRecv,
  IbCastCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  IbCastMakeVDevice,
  IbCastFinalize,
  IbCastSetNetAttr,
};
