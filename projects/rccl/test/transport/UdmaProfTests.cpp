/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "udma_prof.h"

#include <gtest/gtest.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unistd.h>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {

static std::string formatDump() {
  std::string out(1 << 16, '\0');
  size_t n = IbCastUdmaProfFormatJson(&out[0], out.size());
  if (n >= out.size()) {
    out.assign(n + 1, '\0');
    n = IbCastUdmaProfFormatJson(&out[0], out.size());
  }
  out.resize(n);
  return out;
}

TEST(UdmaProf, DisabledIsNoOp) {
  RUN_ISOLATED_TEST("UdmaProfDisabled", []() {
    IbCastUdmaProfResetForTest();
    IbCastUdmaProfRegisterQp((void*)0x1, 0, 42, 0, "ionic0", 0, 1, 1, 0, 0, "peer", 7);
    IbCastUdmaProfAddBytes(0, 0, 1, 1, 4096);
    std::string json = formatDump();
    EXPECT_NE(json.find("\"qps\": ["), std::string::npos);
    EXPECT_EQ(json.find("ionic0"), std::string::npos);
    EXPECT_EQ(json.find("\"bytes\": 4096"), std::string::npos);
  });
}

TEST(UdmaProf, RegisterAndBytesAggregates) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "UdmaProfRegisterAndBytes",
      []() {
        IbCastUdmaProfResetForTest();
        const void* comm = (void*)0xabc;
        IbCastUdmaProfRegisterQp(comm, 0, 100, 0, "ionic0", 0, 1, 1, 3, 0, "hostB", 22);
        IbCastUdmaProfRegisterQp(comm, 3, 102, 0, "ionic0", 0, 1, 1, 3, 1, "hostB", 22);
        IbCastUdmaProfRegisterQp(comm, 1, 101, 0, "ionic0", 1, 1, 1, 3, 1, "hostB", 22);
        IbCastUdmaProfRegisterQp(comm, 2, 200, 1, "ionic1", 0, 0, 0, 1, 0, "hostC", 33);
        IbCastUdmaProfAddBytes(0, 0, 1, 1, 1024);
        IbCastUdmaProfAddBytes(0, 1, 1, 1, 2048);
        IbCastUdmaProfAddBytes(1, 0, 0, 0, 64);

        std::string json = formatDump();
        EXPECT_NE(json.find("\"type\": \"p2p-data\""), std::string::npos);
        EXPECT_NE(json.find("\"type\": \"coll-cts\""), std::string::npos);
        EXPECT_NE(json.find("\"dst\": \"hostB:22\""), std::string::npos);
        EXPECT_NE(json.find("\"dst\": \"hostC:33\""), std::string::npos);
        EXPECT_NE(json.find("\"udma\": \"low\""), std::string::npos);
        EXPECT_NE(json.find("\"udma\": \"high\""), std::string::npos);
        EXPECT_NE(json.find("\"nic\": \"ionic0\""), std::string::npos);
        EXPECT_NE(json.find("\"count\": 2"), std::string::npos);
        EXPECT_NE(json.find("\"bytes\": 1024"), std::string::npos);
        EXPECT_NE(json.find("\"bytes\": 2048"), std::string::npos);
        EXPECT_NE(json.find("\"bytes\": 64"), std::string::npos);

        IbCastUdmaProfUpdateQpn(comm, 0, 999);
        json = formatDump();
        EXPECT_NE(json.find("\"qpn\": 999"), std::string::npos);
        EXPECT_EQ(json.find("\"qpn\": 100"), std::string::npos);
      },
      {{"RCCL_IB_UDMA_PROF", "1"}});
}

TEST(UdmaProf, DumpWritesFile) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "UdmaProfDumpWritesFile",
      []() {
        IbCastUdmaProfResetForTest();
        IbCastUdmaProfRegisterQp((void*)0x2, 0, 7, 0, "nic0", 1, 1, 0, 0, 0, "dst", 9);
        IbCastUdmaProfDump();
        char path[4096];
        IbCastUdmaProfDumpPath(path, sizeof(path));
        FILE* fp = fopen(path, "r");
        ASSERT_NE(fp, nullptr) << path;
        char buf[4096] = {};
        size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
        fclose(fp);
        ASSERT_GT(n, 0u);
        EXPECT_NE(strstr(buf, "p2p-cts"), nullptr);
        EXPECT_NE(strstr(buf, "nic0"), nullptr);
        unlink(path);
      },
      {{"RCCL_IB_UDMA_PROF", "1"}, {"RCCL_IB_UDMA_PROF_DIR", "/tmp"}});
}

} // namespace RcclUnitTesting
