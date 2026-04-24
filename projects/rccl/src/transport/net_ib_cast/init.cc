/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net_ib_cast_common.h"
#include "p2p_resiliency_recovery.h"

extern ncclResult_t pciPathToInt64(char* path, int offset, int minOffset, int64_t* id);
extern int64_t ncclParamIbCastSplitDataOnQps();

NCCL_PARAM(IbCastGidIndex, "IB_GID_INDEX", -1);
NCCL_PARAM(IbCastRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM(IbCastRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
NCCL_PARAM(IbCastTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(IbCastRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbCastPkey, "IB_PKEY", 0);
NCCL_PARAM(IbCastUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbCastSl, "IB_SL", -1);
NCCL_PARAM(IbCastTc, "IB_TC", -1);
NCCL_PARAM(IbCastArThreshold, "IB_AR_THRESHOLD", 8192);
NCCL_PARAM(IbCastPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbCastAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbCastFifoTc, "IB_FIFO_TC", -1);
NCCL_PARAM(IbCastEceEnable,"IB_ECE_ENABLE",1);
NCCL_PARAM(IbCastDataDirect,"IB_DATA_DIRECT",1);
NCCL_PARAM(IbCastQpsPerConn, "IB_QPS_PER_CONNECTION", 2);
RCCL_PARAM(IbCastQpsPerP2p, "IB_QPS_PER_P2P", 0);
NCCL_PARAM(IbCastGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);
RCCL_PARAM(IbCastCtsInlineData, "CTS_INLINE_DATA", -1);
RCCL_PARAM(IbCastCtsOffloadEnabled, "CTS_OFFLOAD_ENABLED", -1);
RCCL_PARAM(IbCastGdrFlushGpuMemNoRelaxedOrdering, "GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 1);

extern int64_t rcclParamAinicRoce();

#define NSEC_PER_USEC           1000ULL
#define NSEC_PER_MSEC           (NSEC_PER_USEC * 1000)
#define NSEC_PER_SEC            (NSEC_PER_MSEC * 1000)
#define NSEC_PER_MIN            (NSEC_PER_SEC * 60)

#define QP_SCHED_RESET_MIN      NSEC_PER_MSEC
#define QP_SCHED_RESET_NEVER    0

#define QP_SCHED_UPDATE_MIN     NSEC_PER_USEC
#define QP_SCHED_UPDATE_MAX     NSEC_PER_MIN

#define QP_SCHED_WEIGHT_NONE    0
#define QP_SCHED_WEIGHT_MIN     QP_SCHED_WEIGHT_NONE
#define QP_SCHED_WEIGHT_MAX     1.0

#define QP_SCHED_DISABLE        0
#define QP_SCHED_ENABLE         1

#define QP_SCHED_ENABLE_DEF             QP_SCHED_ENABLE
#define QP_SCHED_WRR_ENABLE_DEF         QP_SCHED_ENABLE
#define QP_SCHED_RESET_DEF              (NSEC_PER_SEC * 60)
#define QP_SCHED_UPDATE_DEF             (NSEC_PER_USEC * 50)
#define QP_SCHED_WEIGHT_DEF             QP_SCHED_WEIGHT_NONE
#define QP_SCHED_SPLIT_DATA_MIN_DEF     (64 * 1024)
#define QP_SCHED_LOG_DEF                NSEC_PER_SEC

#define QP_SCHED_WEIGHT_ENV_VAR           "RCCL_IB_QP_SCHED_WEIGHT"
#define QP_SCHED_WEIGHT_ENV_VAR_ALIAS     "NCCL_IB_QP_SCHED_WEIGHT"
#define QP_SCHED_LOG_PATH_ENV_VAR         "RCCL_IB_QP_SCHED_LOG_PATH"
#define QP_SCHED_LOG_PATH_ENV_VAR_ALIAS   "NCCL_IB_QP_SCHED_LOG_PATH"

#define QP_SCHED_LOG_FILE_NAME_PREFIX     "cast_log_"

struct ncclIbQpSchedParms castGlobalQpSchedParms;

pthread_mutex_t ncclIbQpSchedParmsLock = PTHREAD_MUTEX_INITIALIZER;
struct ncclIbQpSchedParmsCB stagedSchedParms { ncclNumFuncs };
ncclFunc_t IbCastQpSchedProxyPrevCollType = ncclNumFuncs;
size_t IbCastQpSchedProxyPrevMsgSz;

FILE *IbCastQpSchedLogStream;

RCCL_PARAM_NCCL_ALIAS(IbCastQpSchedEnable, "IB_QP_SCHED_ENABLE", QP_SCHED_ENABLE_DEF);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedWrrEnable, "IB_QP_SCHED_WRR_ENABLE", QP_SCHED_WRR_ENABLE_DEF);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedResetInterval, "IB_QP_SCHED_RESET_INTERVAL", -1);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedUpdateInterval, "IB_QP_SCHED_UPDATE_INTERVAL", -1);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedSplitDataMin, "IB_QP_SCHED_SPLIT_DATA_MIN", -1);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedLogInterval, "IB_QP_SCHED_LOG_INTERVAL", -1);

// default to 0 to disable ooo rq, if set to 1, ooo rq will be enabled or failed
NCCL_PARAM(IbOooRq,"IB_OOO_RQ", 0)

static std::mutex ncclIbMutex;

// With ncclNet_v11_t the NCCL core initializes the network plugin per-communicator
// rather than once for all communicators. However, the internal plugin implementation
// still assumes the plugin is initialized only once across all communicators. The ref
// counter makes sure the plugin internally initializes only once. When per communicator
// context support is added to the plugin the ref counter can be removed.
static int netRefCount;

NCCL_PARAM(IbCastDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbCastMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbCastMergeNics, "IB_MERGE_NICS", 1);
NCCL_PARAM(IbDevicePciOrder, "IB_DEVICE_PCI_ORDER", 1);

extern int64_t ncclParamIbCastArThreshold();

// Returns 0 if this is the path of two VFs of the same physical device
static int ncclIbMatchVfPath(char* path1, char* path2) {
  // Merge multi-port NICs into the same PCI device
  if (ncclParamIbCastMergeVfs()) {
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}

static int ncclIbCompareDevs(const void* dev1, const void* dev2) {
  // Compare devices using the last component of the PCI path.
  // Note: fullPciPath is never NULL but empty if not found by realpath.
  char* path1 = ((struct ncclIbDev*)dev1)->fullPciPath;
  char* path2 = ((struct ncclIbDev*)dev2)->fullPciPath;

  // if a path is empty, order the devices to the back of the list
  if (strlen(path1) == 0 || strlen(path2) == 0) return strlen(path2) - strlen(path1);

  int64_t id1, id2;
  pciPathToInt64(path1, strlen(path1), 0, &id1);
  pciPathToInt64(path2, strlen(path2), 0, &id2);

  return (id1 < id2) ? -1 : ((id1 == id2) ? 0 : 1);
}

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, char* fullPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  // set fullPath to empty if realpath returned NULL
  snprintf(fullPath, PATH_MAX, "%s", p ? p : "");
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (ncclParamIbCastMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
  }
  if (path) {
    *path = p;
  } else {
    free(p);
  }
  return ncclSuccess;
}

static ncclResult_t ncclIbGetRealPort(char* pciPath, int* realPort, int devIdx) {
  *realPort = 0;
  if (pciPath == NULL) return ncclSuccess;
  // Keep the real port aside (the ibv port is always 1 on recent cards)
  // Count only devices before the current device index to assign unique port numbers
  for (int d = 0; d < devIdx; d++) {
    if (ncclIbMatchVfPath(pciPath, IbCastDevs[d].pciPath)) (*realPort)++;
  }
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000, /* NDR */
  200000  /* XDR */
};

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbCastPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

static bool ncclMlx5dvDmaBufCapable(ibv_context *context){
  ncclResult_t res;
  int dev_fail = 0;

  struct ibv_pd* pd;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, context), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1).
  // Check errno after each call since the next call overwrites it.
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  (void)wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/, 0 /* mlx5 flags*/);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  return true;
failure:
  return false;
}

extern int64_t ncclParamIbPrepostReceiveWorkRequests();
extern int64_t ncclParamIbReceiverSideMatchingScheme();

static ncclResult_t ncclIbQueryOooRqSize(struct ibv_context* ibvCtx, const char *devName, uint32_t* oooRqSize) {
  ncclResult_t ret;
  if (!oooRqSize) return ncclInvalidArgument;
  *oooRqSize = 0;

  if (ncclParamIbOooRq() == 0) return ncclSuccess;

  // out-of-order recv prerequisite: device capability
  struct mlx5dv_context dvCtx;
  *oooRqSize = 0;
  dvCtx.comp_mask = MLX5DV_CONTEXT_MASK_OOO_RECV_WRS;
  NCCLCHECKGOTO(wrap_mlx5dv_query_device(ibvCtx, &dvCtx), ret, fail);
  if ((dvCtx.comp_mask & MLX5DV_CONTEXT_MASK_OOO_RECV_WRS) && dvCtx.ooo_recv_wrs_caps.max_rc > 0) {
    *oooRqSize = dvCtx.ooo_recv_wrs_caps.max_rc;
  }


  return ncclSuccess;
fail:
  return ncclInternalError;
}

static ncclResult_t ncclIbGetPciRootFromPath(
    const char* pciPath,
    char* root,
    size_t rootLen
) {
    if (pciPath == NULL || root == NULL || rootLen < 8){
        return ncclInvalidUsage;
    }
    const char* p = strstr(pciPath, "pci");
    while (p != NULL) {
        int domain, bus;
        int chars_read = 0;
        if (sscanf(p, "pci%4x:%2x%n", &domain, &bus, &chars_read) == 2 &&
            chars_read == 10) {
            snprintf(root, rootLen, "%04x:%02x", domain, bus);
            return ncclSuccess;
        }
        p = strstr(p + 1, "pci");
    }
    return ncclInvalidUsage;
}

static int ncclIbGetNumaNodeFromPath(const char* pciPath) {
    if (pciPath == NULL) {
        return -1;
    }
    char numaPath[PATH_MAX];
    if (snprintf(numaPath, sizeof(numaPath), "%s/numa_node", pciPath) >= PATH_MAX) {
        return -1;
    }

    int fd = open(numaPath, O_RDONLY);
    if (fd < 0) return -1;

    char buf[32];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (n <= 0) return -1;
    buf[n] = '\0';

    char* endptr;
    errno = 0;
    long numa = strtol(buf, &endptr, 10);
    if (endptr == buf || errno == ERANGE) {
        return -1;
    }
    return (int)numa;
}

ncclResult_t IbCastMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props) {
  if (ncclParamIbCastMergeNics() == 0 && props->ndevs > 1) {
    WARN("NET/IB : Skipping makeVDevice, Please set NCCL_IB_MERGE_NICS=1");
    return ncclInvalidUsage;
  }

  if (props->ndevs == 0) {
      WARN("NET/IB : Can't make virtual NIC with 0 devices");
      return ncclInvalidUsage;
  }

  if (ncclNMergedIbDevs == MAX_IB_VDEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_VDEVS);
    return ncclInvalidUsage;
  }

  // Always count up number of merged devices
  ncclIbMergedDev* mDev = IbCastMergedDevs + ncclNMergedIbDevs;
  mDev->vProps.ndevs = 0;
  mDev->speed = 0;

  for (int i = 0; i < props->ndevs; i++) {
    ncclIbDev* dev = IbCastDevs + props->devs[i];
    if (mDev->vProps.ndevs == NCCL_IB_MAX_DEVS_PER_NIC) return ncclInvalidUsage;
    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
    mDev->speed += dev->speed;
    // Each successive time, copy the name '+' new name
    if (mDev->vProps.ndevs > 1) {
      snprintf(mDev->devName + strlen(mDev->devName), sizeof(mDev->devName) - strlen(mDev->devName), "+%s", dev->devName);
    // First time, copy the plain name
    } else {
      strncpy(mDev->devName, dev->devName, MAXNAMESIZE);
    }
  }

  // Check link layers
  ncclIbDev* dev0 = IbCastDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    if (props->devs[i] >= ncclNIbDevs) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i], ncclNIbDevs);
      return ncclInvalidUsage;
    }
    ncclIbDev* dev = IbCastDevs + props->devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
        props->devs[0], dev0->devName, dev0->portNum, NCCL_IB_LLSTR(dev0->link), props->devs[i], dev->devName, dev->portNum, NCCL_IB_LLSTR(dev->link));
      return ncclInvalidUsage;
    }
  }

  int numa0 = ncclIbGetNumaNodeFromPath(dev0->pciPath);
  char root0[8];
  ncclIbGetPciRootFromPath(dev0->pciPath, root0, sizeof(root0));
  for (int i = 1; i < props->ndevs; i++) {
    ncclIbDev* dev = IbCastDevs + props->devs[i];
    int numa_i = ncclIbGetNumaNodeFromPath(dev->pciPath);
    if (numa0 >= 0 && numa_i >= 0 && numa_i != numa0) {
      WARN("NET/IB : Merging NICs across NUMA nodes (%s numa=%d, %s numa=%d). "
           "This may significantly reduce performance.",
           dev0->devName, numa0, dev->devName, numa_i);
      break;
    }

    char root_i[8];
    ncclIbGetPciRootFromPath(dev->pciPath, root_i, sizeof(root_i));
    if (strcmp(root_i, root0) != 0) {
      WARN("NET/IB : Merging NICs across PCIe Root Complexes "
           "(%s root=%s, %s root=%s). "
           "GPUDirect RDMA and bandwidth aggregation may be impacted.",
           dev0->devName, root0, dev->devName, root_i);
      break;
    }
  }

  // CTS Offload and CTS Inline are not yet compatible with NIC Fusion
  // (NCCL_IB_MERGE_NICS). Disable them when a multi-NIC vNIC is created.
  if (props->ndevs > 1) {
      if (rcclCtsInlineData) {
        INFO(NCCL_INIT|NCCL_NET, "NET/IB : NIC Fusion (ndevs=%d) - disabling CTS Inline Data (not yet supported with merge)", props->ndevs);
        rcclCtsInlineData = false;
      }
    if (rcclCtsOffloadEnabled) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : NIC Fusion (ndevs=%d) - disabling CTS Offload (not yet supported with merge)", props->ndevs);
      rcclCtsOffloadEnabled = false;
    }
  }

  *d = ncclNMergedIbDevs++;
  INFO(NCCL_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, mDev->devName, mDev->speed, mDev->vProps.ndevs);
  return ncclSuccess;
}

ncclResult_t IbCastMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  std::lock_guard<std::mutex> lock(ncclIbMutex);
  ncclResult_t res = IbCastMakeVDeviceInternal(d, props);
  return res;
}

ncclResult_t IbCastSetNetAttr(void *ctx, ncclNetAttr_t *netAttr) {
  (void)ctx;
  (void)netAttr;
  return ncclSuccess;
}

const char* IbCastProviderName[] = {
  "None",
  "Mlx5",
};

ncclResult_t IbCastQpSchedInitParms(struct ncclIbQpSchedParms *parms) {
  char *str, *logFileName = NULL;
  int val;
  double weight;
  uint64_t nsec;
  ncclResult_t ret = ncclSuccess;

  parms->enable = QP_SCHED_ENABLE_DEF;
  parms->wrrEnable = QP_SCHED_WRR_ENABLE_DEF;
  parms->resetInterval = QP_SCHED_RESET_DEF;
  parms->updateInterval = QP_SCHED_UPDATE_DEF;
  parms->weightNew = QP_SCHED_WEIGHT_DEF;
  parms->splitData = ncclParamIbCastSplitDataOnQps();
  parms->splitDataMin = QP_SCHED_SPLIT_DATA_MIN_DEF;
  parms->logInterval = QP_SCHED_LOG_DEF;
  parms->logEnable = false;
  if (!parms->enable)
    parms->doWrr = false;
  else if (!parms->splitData)
    parms->doWrr = true;
  else if (parms->wrrEnable)
    parms->doWrr = true;
  else
    parms->doWrr = false;

  if (rcclParamIbCastQpSchedEnable()) {
    parms->enable = true;
    INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_ENABLE set to enabled");
  } else {
    parms->enable = false;
    INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_ENABLE set to disabled");
    goto exit;
  }

  if (parms->splitData) {
    if (rcclParamIbQpSchedWrrEnable()) {
      parms->doWrr = true;
      INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_WRR_ENABLE set to enabled");
    }
  } else
    parms->doWrr = true;

  val = (int)rcclParamIbQpSchedResetInterval();
  if (val >= 0) {
    nsec = (val * NSEC_PER_MSEC);
    if (nsec > 0 && nsec < QP_SCHED_RESET_MIN)
      goto getUpdateParm;
    parms->resetInterval = nsec;
    INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_RESET_INTERVAL set to %lu nsec", nsec);
  }

getUpdateParm:
  val = (int)rcclParamIbQpSchedUpdateInterval();
  if (val >= 0) {
    nsec = (val * NSEC_PER_USEC);
    if ((nsec >= QP_SCHED_UPDATE_MIN) && (nsec <= QP_SCHED_UPDATE_MAX)) {
      parms->updateInterval = nsec;
      INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_UPDATE_INTERVAL set to %lu nsec", nsec);
    }
  }

  str = getenv(QP_SCHED_WEIGHT_ENV_VAR);
  if (!str)
    str = getenv(QP_SCHED_WEIGHT_ENV_VAR_ALIAS);
  if (str) {
    weight = atof(str);
    if (weight != QP_SCHED_WEIGHT_NONE) {
      if (weight <= QP_SCHED_WEIGHT_MAX && weight >= QP_SCHED_WEIGHT_MIN) {
        parms->weightNew = weight;
        INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_WEIGHT set to %f", weight);
      }
    }
  }

  val = (int)rcclParamIbQpSchedSplitDataMin();
  if (val > 0) {
    parms->splitDataMin = val;
    INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_SPLIT_DATA_MIN set to %d bytes", val);
  }

  str = getenv(QP_SCHED_LOG_PATH_ENV_VAR);
  if (!str)
    str = getenv(QP_SCHED_LOG_PATH_ENV_VAR_ALIAS);
  if (str != NULL) {
    char hostName[HOST_NAME_MAX + 1], pid[32];
    size_t fileNameLen;

    gethostname(hostName, HOST_NAME_MAX);
    snprintf(pid, sizeof(pid), "%d", getpid());
    fileNameLen = strlen(str) + 1 + strlen(QP_SCHED_LOG_FILE_NAME_PREFIX) +
                  strlen(hostName) + 1 + strlen(pid);
    logFileName = (char *) calloc(1, fileNameLen + 1);
    if (logFileName == NULL) {
      WARN("(IB-CAST) NCCL_IB_QP_SCHED_LOG_PATH: calloc failed");
      goto err_exit;
    }
    snprintf(logFileName, fileNameLen + 1, "%s/%s%s_%s", str, QP_SCHED_LOG_FILE_NAME_PREFIX, hostName, pid);
    IbCastQpSchedLogStream = fopen(logFileName, "w");
    if (IbCastQpSchedLogStream == NULL) {
      WARN("(IB-CAST) NCCL_IB_QP_SCHED_LOG_PATH: fopen failed: %s (logging disabled)", strerror(errno));
      parms->logEnable = false;
      goto exit;
    }
    INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) NCCL_IB_QP_SCHED_LOG_PATH: opened %s", logFileName);
    parms->logEnable = true;

    val = (int)rcclParamIbQpSchedLogInterval();
    if (val >= 0) {
      nsec = (val * NSEC_PER_USEC);
      if ((nsec >= QP_SCHED_UPDATE_MIN) && (nsec <= QP_SCHED_UPDATE_MAX)) {
        parms->logInterval = nsec;
        INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_LOG_INTERVAL set to %lu nsec", nsec);
      }
    }
  }

  INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) NCCL_IB_QPS_PER_CONNECTION set to %d", ncclParamIbCastQpsPerConn());

exit:
  free(logFileName);
  return ret;

err_exit:
  ret = ncclInternalError;
  goto exit;
}

ncclResult_t IbCastFinalizeDevices(void) {
  netRefCount--;
  return ncclSuccess;
}

extern int64_t ncclIbArThreshold;
ncclResult_t IbCastInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  if (netRefCount++) return ret;
  ncclProfilerFunction = profFunction;
  if (ncclParamIbCastDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }
  if(wrap_mlx5dv_symbols() != ncclSuccess) { INFO(NCCL_NET, "NET/IB : Failed to open mlx5dv symbols. Advance features like CX-8 Direct-NIC will be disabled."); }
  if(wrap_ionicdv_symbols() != ncclSuccess) {
    INFO(NCCL_NET, "NET/IB : Failed to open ionicdv symbols. Advance features like AINIC UD load balancing will be disabled.");
  }

  if (ncclNIbDevs == -1) {
    std::lock_guard<std::mutex> lock(ncclIbMutex);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      int nIpIfs = 0;
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      NCCLCHECK(ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs = 0;
      struct ibv_device** devices = NULL;

      // Check if user defined which IB device:port to use
      const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = ncclInternalError; goto fail; }

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context = NULL;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        char dataDirectDevicePath[PATH_MAX] = "/sys";
        int devCount = /*undefined*/-1, devOffset = 0;

        uint32_t oooRqSize = 0;
        enum ncclIbProvider ibProvider = wrap_mlx5dv_is_supported(devices[d]) ? IB_PROVIDER_MLX5 : IB_PROVIDER_NONE;
        if (ibProvider == IB_PROVIDER_MLX5 && ncclParamIbOooRq()) {
          NCCLCHECKGOTO(ncclIbQueryOooRqSize(context, devices[d]->name, &oooRqSize), ret, fail);
        }

        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
            struct ibv_port_attr portAttr;
            if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
              WARN("NET/IB : Unable to query port_num %d", port_num);
              continue;
            }
            if (portAttr.state != IBV_PORT_ACTIVE) continue;
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

            // check against user specified HCAs/ports
            if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
              continue;
            }

            // check for mlx5 data direct support only once for a each device
            if (devCount == -1) {
              devCount = 1;
              devOffset = 0;
              if (ncclParamIbCastDataDirect() > 0 && ibProvider == IB_PROVIDER_MLX5 && ncclMlx5dvDmaBufCapable(context)) {
                int pathLen = strlen(dataDirectDevicePath);
                ncclResult_t res = wrap_mlx5dv_get_data_direct_sysfs_path(context, dataDirectDevicePath + pathLen, sizeof(dataDirectDevicePath) - pathLen);
                if (res == ncclSuccess) {
                  // data direct devices are exposed twice: with the C2C + PCIe link and with the data direct link
                  devCount = 2;
                  // by default only expose the data direct NIC (devOffset = 1), unless set to 2 by the user
                  devOffset = (ncclParamIbCastDataDirect() == 2) ? 0 : 1;
                  INFO(NCCL_INIT | NCCL_NET, "NET/IB: Data Direct DMA Interface is detected for device %s", devices[d]->name);
                } else if (res == ncclInvalidArgument) {
                  TRACE(NCCL_NET, "NET/IB: Device %s does not support Data Direct DMA.", devices[d]->name);
                } else {
                  WARN("NET/IB: Error in mlx5dv_get_data_direct_sysfs_path with device %s", devices[d]->name);
                  return res;
                }
              }
            }
            for (int dev = devOffset; dev < devCount; ++dev) {
              IbCastDevs[ncclNIbDevs].device = d;
              IbCastDevs[ncclNIbDevs].ibProvider = ibProvider;
              IbCastDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
              IbCastDevs[ncclNIbDevs].portAttr = portAttr;
              IbCastDevs[ncclNIbDevs].portNum = port_num;
              IbCastDevs[ncclNIbDevs].link = portAttr.link_layer;
              if (portAttr.active_speed_ex) {
                // A non-zero active_speed_ex indicates XDR rate (0x100) or higher
                IbCastDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed_ex) * ncclIbWidth(portAttr.active_width);
              } else {
                IbCastDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
              }
              IbCastDevs[ncclNIbDevs].context = context;
              IbCastDevs[ncclNIbDevs].pdRefs = 0;
              IbCastDevs[ncclNIbDevs].pd = NULL;
              // for dev==1 (data direct device), pciPath is given by mlx5
              strncpy(IbCastDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
              NCCLCHECKGOTO(ncclIbGetPciPath(IbCastDevs[ncclNIbDevs].devName, (dev == 1) ? NULL : &IbCastDevs[ncclNIbDevs].pciPath, IbCastDevs[ncclNIbDevs].fullPciPath), ret, fail);
              if (dev == 1) {
                snprintf(IbCastDevs[ncclNIbDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
                NCCLCHECK(ncclCalloc(&IbCastDevs[ncclNIbDevs].pciPath, PATH_MAX));
                strncpy(IbCastDevs[ncclNIbDevs].pciPath, dataDirectDevicePath, PATH_MAX);
                IbCastDevs[ncclNIbDevs].capsProvider.mlx5.dataDirect = 1;
              }

              IbCastDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
              IbCastDevs[ncclNIbDevs].oooRqSize = oooRqSize;
              IbCastDevs[ncclNIbDevs].mrCache.capacity = 0;
              IbCastDevs[ncclNIbDevs].mrCache.population = 0;
              IbCastDevs[ncclNIbDevs].mrCache.slots = NULL;
              NCCLCHECK(IbCastStatsInit(&IbCastDevs[ncclNIbDevs].stats));

              // Enable ADAPTIVE_ROUTING by default on IB networks
              // But allow it to be overloaded by an env parameter
              IbCastDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
              if (ncclParamIbCastAdaptiveRouting() != -2) IbCastDevs[ncclNIbDevs].ar = ncclParamIbCastAdaptiveRouting();


              INFO(NCCL_NET, "NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d oooRqSize=%d", d, devices[d]->name, devices[d]->dev_name,
                   IbCastDevs[ncclNIbDevs].portNum, NCCL_IB_LLSTR(portAttr.link_layer), IbCastProviderName[IbCastDevs[ncclNIbDevs].ibProvider], IbCastDevs[ncclNIbDevs].speed, context,
                   IbCastDevs[ncclNIbDevs].pciPath, IbCastDevs[ncclNIbDevs].ar, IbCastDevs[ncclNIbDevs].oooRqSize);

              pthread_create(&IbCastAsyncThread, NULL, IbCastAsyncThreadMain, IbCastDevs + ncclNIbDevs);

              ncclNIbDevs++;
              nPorts++;
            }
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
      }

      if (devices && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; }
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }
    // Determine whether RELAXED_ORDERING is enabled and possible
    ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();

    // Default value for ncclIbArThreshold is 8192
    if (ncclParamIbCastArThreshold() != -2) {
      if (ncclParamIbOooRq()) {
        INFO(NCCL_NET, "NET/IB: OOO RQ is enabled, AR threshold will be ignored.");
      } else {
        ncclIbArThreshold = ncclParamIbCastArThreshold();  // set explicitly by user
      }
    }
    // sort devices to ensure a consistent order across nodes
    if (ncclParamIbDevicePciOrder()) qsort(IbCastDevs, ncclNIbDevs, sizeof(struct ncclIbDev), ncclIbCompareDevs);
    // Once sorted, get the realPort ID and create the virtual devices.
    // Doing it after sorting ensures that devices will have consistent realPort ids across nodes.
    char line[2048] = "";
    for (int d = 0; d < ncclNIbDevs; d++) {
      NCCLCHECKGOTO(ncclIbGetRealPort(IbCastDevs[d].pciPath, &IbCastDevs[d].realPort, d), ret, fail);
      snprintf(line + strlen(line), sizeof(line) - strlen(line), " [%d]%s:%d/%s", d, IbCastDevs[d].devName, IbCastDevs[d].portNum, NCCL_IB_LLSTR(IbCastDevs[d].link));

      // Add this plain physical device to the list of virtual devices (after sorting)
      int vDev;
      ncclNetVDeviceProps_t vProps = {0};
      vProps.ndevs = 1;
      vProps.devs[0] = d;
      NCCLCHECK(IbCastMakeVDeviceInternal(&vDev, &vProps));
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT | NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "", ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));

    // Initialize QP scheduling parameters
    IbCastQpSchedInitParms(&castGlobalQpSchedParms);

    // Detect AINIC RoCEv2 and configure CTS features
    rcclAinicRoce = (rcclParamAinicRoce() > 0);
    if (rcclAinicRoce) {
      int ctsOffload = rcclParamIbCastCtsOffloadEnabled();
      int ctsInline  = rcclParamIbCastCtsInlineData();
      rcclCtsOffloadEnabled = (ctsOffload < 0) ? true  : (bool)ctsOffload;
      rcclCtsInlineData     = (ctsInline  < 0) ? false : (bool)ctsInline;
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : AINIC RoCEv2 detected. CTS offload=%d, CTS inline=%d",
           rcclCtsOffloadEnabled, rcclCtsInlineData);
    }
    ncclIbGdrFlushDisable = ncclParamIbCastGdrFlushDisable();
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t IbCastInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(IbCastInitDevices(logFunction, profFunction));
  NCCLCHECK(ncclIbPortRecoveryThreadStart());
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = config->trafficClass;
  *ctx = (void *)netCommConfig;
  return ret;
}

ncclResult_t IbCastDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

ncclResult_t IbCastGetPhysProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbDev* ibDev = IbCastDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (IbCastGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (IbCastDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->forceFlush = 0;
  if (ibDev->capsProvider.mlx5.dataDirect) {
    props->forceFlush = 1;
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  /* props->railId and props->planeId: NCCL 2.30.4 fields, not in RCCL's ncclNetProperties_v11_t */
  return ncclSuccess;
}

ncclResult_t IbCastGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, ncclNMergedIbDevs);
    return ncclInvalidUsage;
  }
  struct ncclIbMergedDev* mergedDev = IbCastMergedDevs + dev;
  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  NCCLCHECK(IbCastGetPhysProperties(mergedDev->vProps.devs[0], props));
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;
  memcpy(&props->vProps, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));
  return ncclSuccess;
}

ncclResult_t IbCastFinalize(void* ctx) {
  free(ctx);
  NCCLCHECK(ncclIbPortRecoveryThreadStop());
  return IbCastFinalizeDevices();
}
