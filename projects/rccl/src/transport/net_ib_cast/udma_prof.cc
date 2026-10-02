/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "udma_prof.h"

#include <pthread.h>
#include <atomic>
#include <map>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <vector>

RCCL_PARAM(IbCastUdmaProf, "IB_UDMA_PROF", 0);

#define IB_CAST_UDMA_PROF_UDMA_N 3
#define IB_CAST_UDMA_PROF_TYPE_N 4

struct IbCastUdmaProfQp {
  const void* commKey;
  int qpIndex;
  uint32_t qpn;
  int ibDevN;
  char nic[64];
  int8_t udma;
  int isP2p;
  int isDataQp;
  int channelId;
  int qpIndexInDev;
  char destHost[IB_CAST_UDMA_PROF_HOST_MAX];
  int destPid;
};

static std::mutex g_lock;
static std::mutex g_dumpLock;
static std::once_flag g_atexitOnce;
static std::vector<IbCastUdmaProfQp> g_qps;
static std::atomic<uint64_t> g_bytes[IB_CAST_UDMA_PROF_MAX_DEVS][IB_CAST_UDMA_PROF_UDMA_N][IB_CAST_UDMA_PROF_TYPE_N];
static char g_nicName[IB_CAST_UDMA_PROF_MAX_DEVS][64];
static bool g_nicSeen[IB_CAST_UDMA_PROF_MAX_DEVS];

static int udmaIndex(int8_t udma) {
  if (udma < 0) return 2;
  return udma ? 1 : 0;
}

static int typeIndex(int isP2p, int isDataQp) {
  return (isP2p ? 0 : 2) + (isDataQp ? 0 : 1);
}

static const char* udmaName(int8_t udma) {
  if (udma < 0) return "none";
  return udma ? "high" : "low";
}

static const char* typeName(int isP2p, int isDataQp) {
  if (isP2p) return isDataQp ? "p2p-data" : "p2p-cts";
  return isDataQp ? "coll-data" : "coll-cts";
}

static void jsonEscape(FILE* fp, const char* s) {
  fputc('"', fp);
  if (s == nullptr) {
    fputc('"', fp);
    return;
  }
  for (const unsigned char* p = (const unsigned char*)s; *p; p++) {
    if (*p == '"' || *p == '\\') {
      fputc('\\', fp);
      fputc(*p, fp);
    } else if (*p == '\n') {
      fputs("\\n", fp);
    } else {
      fputc(*p, fp);
    }
  }
  fputc('"', fp);
}

void IbCastUdmaProfLocalId(char* host, size_t hostLen, int* pid) {
  if (host && hostLen) {
    memset(host, 0, hostLen);
    if (gethostname(host, hostLen) != 0) {
      strncpy(host, "unknown", hostLen - 1);
    }
    host[hostLen - 1] = '\0';
  }
  if (pid) *pid = (int)getpid();
}

static void localId(char* host, size_t hostLen, int* pid) {
  IbCastUdmaProfLocalId(host, hostLen, pid);
}

void IbCastUdmaProfDumpPath(char* buf, size_t cap) {
  if (buf == nullptr || cap == 0) return;
  char host[IB_CAST_UDMA_PROF_HOST_MAX];
  int pid = 0;
  localId(host, sizeof(host), &pid);
  const char* dir = getenv("RCCL_IB_UDMA_PROF_DIR");
  if (dir == nullptr || dir[0] == '\0') dir = ".";
  snprintf(buf, cap, "%s/rccl_udma_prof.%s.%d.json", dir, host, pid);
}

static void writeJson(FILE* fp) {
  char host[IB_CAST_UDMA_PROF_HOST_MAX];
  int pid = 0;
  localId(host, sizeof(host), &pid);

  std::vector<IbCastUdmaProfQp> qps;
  char nicName[IB_CAST_UDMA_PROF_MAX_DEVS][64];
  bool nicSeen[IB_CAST_UDMA_PROF_MAX_DEVS];
  uint64_t bytes[IB_CAST_UDMA_PROF_MAX_DEVS][IB_CAST_UDMA_PROF_UDMA_N][IB_CAST_UDMA_PROF_TYPE_N];
  {
    std::lock_guard<std::mutex> lock(g_lock);
    qps = g_qps;
    memcpy(nicName, g_nicName, sizeof(nicName));
    memcpy(nicSeen, g_nicSeen, sizeof(nicSeen));
  }
  for (int d = 0; d < IB_CAST_UDMA_PROF_MAX_DEVS; d++) {
    for (int u = 0; u < IB_CAST_UDMA_PROF_UDMA_N; u++) {
      for (int t = 0; t < IB_CAST_UDMA_PROF_TYPE_N; t++) {
        bytes[d][u][t] = g_bytes[d][u][t].load(std::memory_order_relaxed);
      }
    }
  }

  char srcId[IB_CAST_UDMA_PROF_HOST_MAX + 16];
  snprintf(srcId, sizeof(srcId), "%s:%d", host, pid);

  fputs("{\n  \"source\": ", fp);
  jsonEscape(fp, srcId);
  fputs(",\n  \"qps\": [\n", fp);
  for (size_t i = 0; i < qps.size(); i++) {
    const IbCastUdmaProfQp& q = qps[i];
    char dstId[IB_CAST_UDMA_PROF_HOST_MAX + 16];
    snprintf(dstId, sizeof(dstId), "%s:%d", q.destHost, q.destPid);
    fputs("    {\"src\": ", fp);
    jsonEscape(fp, srcId);
    fputs(", \"dst\": ", fp);
    jsonEscape(fp, dstId);
    fputs(", \"nic\": ", fp);
    jsonEscape(fp, q.nic);
    fprintf(fp,
            ", \"ibDevN\": %d, \"udma\": \"%s\", \"type\": \"%s\", \"qpn\": %u, "
            "\"channelId\": %d, \"qpIndexInDev\": %d}",
            q.ibDevN, udmaName(q.udma), typeName(q.isP2p, q.isDataQp), q.qpn, q.channelId,
            q.qpIndexInDev);
    fputs(i + 1 < qps.size() ? ",\n" : "\n", fp);
  }
  fputs("  ],\n  \"qp_counts\": [\n", fp);

  std::map<std::string, int> counts;
  for (const IbCastUdmaProfQp& q : qps) {
    char dstId[IB_CAST_UDMA_PROF_HOST_MAX + 16];
    snprintf(dstId, sizeof(dstId), "%s:%d", q.destHost, q.destPid);
    char key[320];
    snprintf(key, sizeof(key), "%s\n%s\n%s\n%s", q.nic, udmaName(q.udma),
             typeName(q.isP2p, q.isDataQp), dstId);
    counts[key]++;
  }
  size_t nAgg = counts.size();
  size_t aggI = 0;
  for (const auto& kv : counts) {
    char nic[64] = {}, udma[16] = {}, type[16] = {}, dst[IB_CAST_UDMA_PROF_HOST_MAX + 16] = {};
    sscanf(kv.first.c_str(), "%63[^\n]\n%15[^\n]\n%15[^\n]\n%79[^\n]", nic, udma, type, dst);
    fputs("    {\"nic\": ", fp);
    jsonEscape(fp, nic);
    fputs(", \"udma\": ", fp);
    jsonEscape(fp, udma);
    fputs(", \"type\": ", fp);
    jsonEscape(fp, type);
    fputs(", \"dst\": ", fp);
    jsonEscape(fp, dst);
    fprintf(fp, ", \"count\": %d}", kv.second);
    fputs(++aggI < nAgg ? ",\n" : "\n", fp);
  }

  fputs("  ],\n  \"bytes\": [\n", fp);
  int nByte = 0;
  for (int d = 0; d < IB_CAST_UDMA_PROF_MAX_DEVS; d++) {
    for (int u = 0; u < IB_CAST_UDMA_PROF_UDMA_N; u++) {
      for (int t = 0; t < IB_CAST_UDMA_PROF_TYPE_N; t++) {
        if (bytes[d][u][t] == 0) continue;
        nByte++;
      }
    }
  }
  int byteI = 0;
  const char* typeNames[IB_CAST_UDMA_PROF_TYPE_N] = {"p2p-data", "p2p-cts", "coll-data", "coll-cts"};
  const char* udmaNames[IB_CAST_UDMA_PROF_UDMA_N] = {"low", "high", "none"};
  for (int d = 0; d < IB_CAST_UDMA_PROF_MAX_DEVS; d++) {
    for (int u = 0; u < IB_CAST_UDMA_PROF_UDMA_N; u++) {
      for (int t = 0; t < IB_CAST_UDMA_PROF_TYPE_N; t++) {
        if (bytes[d][u][t] == 0) continue;
        char nic[64] = {};
        if (nicSeen[d] && nicName[d][0]) strncpy(nic, nicName[d], sizeof(nic) - 1);
        else snprintf(nic, sizeof(nic), "ibdev%d", d);
        fputs("    {\"nic\": ", fp);
        jsonEscape(fp, nic);
        fprintf(fp, ", \"ibDevN\": %d, \"udma\": \"%s\", \"type\": \"%s\", \"bytes\": %llu}", d,
                udmaNames[u], typeNames[t], (unsigned long long)bytes[d][u][t]);
        fputs(++byteI < nByte ? ",\n" : "\n", fp);
      }
    }
  }
  fputs("  ]\n}\n", fp);
}

static void atExitDump(void) {
  if (IbCastUdmaProfEnabled()) IbCastUdmaProfDump();
}

static void ensureAtExit(void) {
  std::call_once(g_atexitOnce, []() { atexit(atExitDump); });
}

void IbCastUdmaProfRegisterQp(const void* commKey, int qpIndex, uint32_t qpn, int ibDevN,
                              const char* nicName, int8_t udma, int isP2p, int isDataQp,
                              int channelId, int qpIndexInDev, const char* destHost, int destPid) {
  if (!IbCastUdmaProfEnabled()) return;
  ensureAtExit();

  IbCastUdmaProfQp rec = {};
  rec.commKey = commKey;
  rec.qpIndex = qpIndex;
  rec.qpn = qpn;
  rec.ibDevN = ibDevN;
  rec.udma = udma;
  rec.isP2p = isP2p ? 1 : 0;
  rec.isDataQp = isDataQp ? 1 : 0;
  rec.channelId = channelId;
  rec.qpIndexInDev = qpIndexInDev;
  rec.destPid = destPid;
  if (nicName) strncpy(rec.nic, nicName, sizeof(rec.nic) - 1);
  if (destHost) strncpy(rec.destHost, destHost, sizeof(rec.destHost) - 1);

  std::lock_guard<std::mutex> lock(g_lock);
  g_qps.push_back(rec);
  if (ibDevN >= 0 && ibDevN < IB_CAST_UDMA_PROF_MAX_DEVS) {
    g_nicSeen[ibDevN] = true;
    if (nicName && g_nicName[ibDevN][0] == '\0') {
      strncpy(g_nicName[ibDevN], nicName, sizeof(g_nicName[ibDevN]) - 1);
    }
  }
}

void IbCastUdmaProfUpdateQpn(const void* commKey, int qpIndex, uint32_t newQpn) {
  if (!IbCastUdmaProfEnabled()) return;
  std::lock_guard<std::mutex> lock(g_lock);
  for (IbCastUdmaProfQp& q : g_qps) {
    if (q.commKey == commKey && q.qpIndex == qpIndex) {
      q.qpn = newQpn;
    }
  }
}

void IbCastUdmaProfAddBytes(int ibDevN, int8_t udma, int isP2p, int isDataQp, uint64_t bytes) {
  if (!IbCastUdmaProfEnabled() || bytes == 0) return;
  if (ibDevN < 0 || ibDevN >= IB_CAST_UDMA_PROF_MAX_DEVS) return;
  g_bytes[ibDevN][udmaIndex(udma)][typeIndex(isP2p, isDataQp)].fetch_add(bytes, std::memory_order_relaxed);
}

void IbCastUdmaProfDump(void) {
  if (!IbCastUdmaProfEnabled()) return;
  ensureAtExit();
  std::lock_guard<std::mutex> dumpLock(g_dumpLock);
  char path[4096];
  char tmpPath[4112];
  IbCastUdmaProfDumpPath(path, sizeof(path));
  snprintf(tmpPath, sizeof(tmpPath), "%s.tmp", path);
  FILE* fp = fopen(tmpPath, "w");
  if (fp == nullptr) return;
  writeJson(fp);
  fclose(fp);
  rename(tmpPath, path);
}

size_t IbCastUdmaProfFormatJson(char* buf, size_t cap) {
  char* mem = nullptr;
  size_t memsz = 0;
  FILE* fp = open_memstream(&mem, &memsz);
  if (fp == nullptr) return 0;
  writeJson(fp);
  fclose(fp);
  if (buf && cap) {
    size_t n = memsz < cap ? memsz : cap - 1;
    memcpy(buf, mem, n);
    buf[n] = '\0';
  }
  free(mem);
  return memsz;
}

void IbCastUdmaProfResetForTest(void) {
  std::lock_guard<std::mutex> lock(g_lock);
  g_qps.clear();
  memset(g_nicName, 0, sizeof(g_nicName));
  memset(g_nicSeen, 0, sizeof(g_nicSeen));
  for (int d = 0; d < IB_CAST_UDMA_PROF_MAX_DEVS; d++) {
    for (int u = 0; u < IB_CAST_UDMA_PROF_UDMA_N; u++) {
      for (int t = 0; t < IB_CAST_UDMA_PROF_TYPE_N; t++) {
        g_bytes[d][u][t].store(0, std::memory_order_relaxed);
      }
    }
  }
}
