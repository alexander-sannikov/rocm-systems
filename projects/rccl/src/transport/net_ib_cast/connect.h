/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_CONNECT_H_
#define NET_IB_CONNECT_H_

#include "net_ib_cast_common.h"
#include "ibvwrap.h"

struct IbCastQpCreateAttr {
  void* qpContext;
  enum ibv_qp_type type;
  bool oooRq;
  struct ibv_cq* cq;
  struct ibv_pd* pd;
  uint32_t maxRecvWorkRequest;
  uint32_t maxSendWorkRequest;
};

// Per-QP connection metatdata
struct IbCastQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;

  // The index of the device on which the QP was created. Allows the sender and
  // receiver side to have asymmetric device configuration, meaning the sender
  // and receiver can use different number of devices.
  int devIndex;
};

struct IbCastResiliencyInfo {
  // QPs used for probing of data transfers in case of QP/device failures.
  struct IbCastQpInfo probingQpsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
  // QPs used for recovery protocol after QP/device failures.
  struct IbCastQpInfo portRecoveryQpsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Structure used to hold information needed to establish the communication
// between the sender and receiver.
// The structure is populated during the connection establishment phase and
// populated by each side of the connection before being sent to the remote
// peer. The remote peer uses the information passed to it from its peer to
// create and initialize its local resources.
struct IbCastConnectionMetadata {
  struct IbCastQpInfo qpInfo[NCCL_IB_MAX_QPS];
  struct IbCastResiliencyInfo resiliencyInfo;
  struct IbCastDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  // An address for a registered memory to be accessed by the peer. The address
  // can be accessed using RDMA using the key specified in IbCastDevInfo::rkey.
  // The sender side gets in this member, from the receiver, the address of the
  // memory to which the sender writes the sizes of the data transfers that
  // the sender sends.
  // The receiver side gets in this member, from the sender, the address of the
  // memory to which the receiver writes the CTS messages.
  uint64_t addr;
  int ndevs;
  int tc;
  int sl;
  int isP2p;
};

enum IbCastCommState {
  IbCastCommStateStart = 0,
  IbCastCommStateConnect = 1,
  IbCastCommStateAccept = 3,
  IbCastCommStateSend = 4,
  IbCastCommStateRecv = 5,
  IbCastCommStateConnecting = 6,
  IbCastCommStateConnected = 7,
  IbCastCommStatePendingReady = 8,
  IbCastCommStateSendDevList = 9,
  IbCastCommStateRecvDevList = 10,
};

struct IbCastCommStage {
  enum IbCastCommState state;
  int offset;
  void* buffer;
  void* comm;
};

struct IbCastHandle {
  union ncclSocketAddress connectAddr;
  uint64_t magic;
  struct IbCastCommStage stage;
  int isP2p;
};

ncclResult_t IbCastQpCreate(struct IbCastQp* qp, struct IbCastQpCreateAttr* createQpAttrs);
ncclResult_t IbCastQpInit(struct IbCastQp* qp);
ncclResult_t IbCastQpRtr(struct IbCastQp* qp);
ncclResult_t IbCastQpRts(struct IbCastQp* qp);
ncclResult_t IbCastQpReset(struct IbCastQp* qp);
ncclResult_t IbCastQpError(struct IbCastQp* qp);

ncclResult_t IbCastCreateQp(uint8_t ib_port, struct IbCastNetCommDevBase* base,
                             int access_flags, void* qp_context, struct IbCastQp* qp,
                             int channel_id, bool data_qp, int8_t cts_qp_slot);
ncclResult_t IbCastRtrQp(struct ibv_qp* qp, struct IbCastGidInfo* sGidInfo, uint32_t dest_qp_num,
                          struct IbCastDevInfo* info, bool fifoTc, int tc, int sl);
ncclResult_t IbCastRtsQp(struct ibv_qp* qp);

ncclResult_t IbCastPostReceiveWorkRequestsOnQp(struct IbCastRecvComm* recvComm, IbCastQp* dataQp);

#endif // NET_IB_CONNECT_H_
