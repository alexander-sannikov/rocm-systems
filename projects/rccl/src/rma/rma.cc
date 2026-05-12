/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "rma/rma.h"

static bool isLsaAccessible(struct ncclComm* comm, int rank) {
  for (int i = 0; i < comm->devrState.lsaSize; i++) {
    if (comm->devrState.lsaRankList[i] == rank) {
      return true;
    }
  }
  return false;
}

ncclResult_t ncclRmaWaitSignal(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // RCCL: CE RMA path excluded — proxy path only
  if (plan->rmaArgs->nRmaTasksProxy > 0) {
    NCCLCHECKGOTO(ncclRmaProxyWaitLaunch(comm, plan, stream), ret, fail);
  }
  // RCCL: nRmaTasksCe branch excluded (no CE RMA path)

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclRmaPut(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // RCCL: CE RMA path excluded — proxy path only
  if (plan->rmaArgs->nRmaTasksProxy > 0) {
    NCCLCHECKGOTO(ncclRmaProxyPutLaunch(comm, plan, stream), ret, fail);
  }
  // RCCL: nRmaTasksCe branch excluded (no CE RMA path)

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchRma(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t stream = comm->planner.streams->stream;

  switch (plan->rmaArgs->func) {
    case ncclFuncPutSignal:
      NCCLCHECKGOTO(ncclRmaPut(comm, plan, stream), ret, fail);
      break;
    case ncclFuncSignal:
      NCCLCHECKGOTO(ncclRmaPut(comm, plan, stream), ret, fail);
      break;
    case ncclFuncWaitSignal:
      NCCLCHECKGOTO(ncclRmaWaitSignal(comm, plan, stream), ret, fail);
      break;
    default:
      ret = ncclInvalidUsage;
  }

exit:
  return ret;
fail:
  goto exit;
}

static inline bool isRmaPutOrSignal(ncclFunc_t func) {
  return (func == ncclFuncPutSignal || func == ncclFuncSignal);
}

// Check if two RMA tasks can be batched together
static inline bool canBatchRmaTasks(struct ncclTaskRma* task1, struct ncclTaskRma* task2) {
  // Check if the tasks are in the same context
  if (task1->ctx != task2->ctx) return false;

  // Check if the tasks are the same function
  if (task1->func == task2->func) return true;

  // Put/Signal tasks can be batched together
  if (isRmaPutOrSignal(task1->func) && isRmaPutOrSignal(task2->func)) {
    return true;
  }

  return false;
}

// Schedule comm->planner RMA tasks to the plan and split the RMA tasks into CE and Proxy tasks
// Then seek opportunities to batch tasks, batching checked for consecutive operations targeting the same context
// - ncclFuncWaitSignal does not perform further batching as the API can already batch waitSignal from multiple peers
// - Consecutive put/signal operation can be batched into the same plan
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  int* peersProxy = nullptr;
  int* nsignalsProxy = nullptr;
  struct ncclKernelPlanner* planner = &comm->planner;

  // Find the first non-empty context queue
  int ctx = -1;
  for (int i = 0; i < comm->config.numRmaCtx; i++) {
    if (!ncclIntruQueueEmpty(&planner->rmaTaskQueues[i])) {
      ctx = i;
      break;
    }
  }

  // No RMA tasks to schedule
  if (ctx == -1) return ncclSuccess;

  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* ctxQueue = &planner->rmaTaskQueues[ctx];

  // Get the first task to determine the operation category
  struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);

  // Initialize plan
  plan->isRma = true;
  plan->rmaArgs = ncclMemoryStackAlloc<struct ncclRmaArgs>(&comm->memScoped);
  plan->rmaArgs->ctx = ctx;
  plan->rmaArgs->func = firstTask->func;
  plan->rmaArgs->nRmaTasks = 0;
  plan->rmaArgs->nRmaTasksProxy = 0;
  // RCCL: nRmaTasksCe excluded (no CE RMA path)

  // WaitSignal tasks
  if (firstTask->func == ncclFuncWaitSignal) {
    // RCCL: CE path excluded — all peers go to proxy
    NCCLCHECKGOTO(ncclCalloc(&peersProxy, firstTask->npeers), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&nsignalsProxy, firstTask->npeers), ret, fail);

    int npeersProxy = 0;

    // Route all peers to proxy (CE path excluded in RCCL)
    for (int i = 0; i < firstTask->npeers; i++) {
      peersProxy[npeersProxy] = firstTask->peers[i];
      nsignalsProxy[npeersProxy] = firstTask->nsignals[i];
      npeersProxy++;
    }

    // Initialize the Proxy task
    if (npeersProxy > 0) {
      struct ncclTaskRma* waitSignalTaskProxy = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
      waitSignalTaskProxy->func = ncclFuncWaitSignal;
      waitSignalTaskProxy->ctx = firstTask->ctx;
      waitSignalTaskProxy->signalMode = firstTask->signalMode;
      waitSignalTaskProxy->peers = peersProxy;
      waitSignalTaskProxy->nsignals = nsignalsProxy;
      waitSignalTaskProxy->npeers = npeersProxy;
      ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, waitSignalTaskProxy);
      plan->rmaArgs->nRmaTasksProxy = 1;
    } else {
      free(peersProxy);
      peersProxy = nullptr;
      free(nsignalsProxy);
      nsignalsProxy = nullptr;
      plan->rmaArgs->nRmaTasksProxy = 0;
    }

    plan->rmaArgs->nRmaTasks = (npeersProxy > 0 ? 1 : 0);
    planner->nTasksRma -= 1;
    // Free the original WaitSignal task (now in Proxy task)
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
  }
  // Put/Signal tasks — RCCL: CE path excluded, all go to proxy
  else {
    plan->rmaArgs->nRmaTasks = 1;
    plan->rmaArgs->nRmaTasksProxy = 1;
    // RCCL: nRmaTasksCe always 0 (CE excluded)

    ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, firstTask);
    planner->nTasksRma -= 1;

    // Batch consecutive tasks from the same context that match operation category
    while (!ncclIntruQueueEmpty(ctxQueue)) {
      struct ncclTaskRma* task = ncclIntruQueueHead(ctxQueue);

      // Check if this task can be batched with the first task
      if (!canBatchRmaTasks(firstTask, task)) {
        break;
      }

      // RCCL: CE path excluded — all tasks to proxy
      ncclIntruQueueDequeue(ctxQueue);
      ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, task);
      plan->rmaArgs->nRmaTasksProxy++;
      plan->rmaArgs->nRmaTasks++;
      planner->nTasksRma -= 1;
    }
  }

  INFO(NCCL_COLL, "scheduleRmaTasksToPlan: rank=%d ctx=%d func=%d nRmaTasks=%d nRmaTasksProxy=%d (CE excluded)",
    comm->rank, ctx, plan->rmaArgs->func, plan->rmaArgs->nRmaTasks, plan->rmaArgs->nRmaTasksProxy);

exit:
  return ret;
fail:
  free(peersProxy);
  free(nsignalsProxy);
  goto exit;
}
