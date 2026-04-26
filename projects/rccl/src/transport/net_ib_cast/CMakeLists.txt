# Transport sources — relative to PROJECT_SOURCE_DIR, as required by the SRC_FILES hipify loop
set(NET_IB_CAST_SOURCES
    src/transport/net_ib_cast/common.cc
    src/transport/net_ib_cast/connect.cc
    src/transport/net_ib_cast/gdr.cc
    src/transport/net_ib_cast/init.cc
    src/transport/net_ib_cast/p2p.cc
    src/transport/net_ib_cast/p2p_resiliency.cc
    src/transport/net_ib_cast/p2p_resiliency_recovery.cc
    src/transport/net_ib_cast/reg.cc
    src/transport/net_ib_cast/gin.cc
    # gin_host_gdaki.cc excluded: requires DOCA SDK + CUDA, not available in ROCm-only builds
    src/transport/net_ib_cast/deps/gdaki/gin_host_gdaki_stub.cc
)

add_subdirectory(gdaki)

# Include directories for net_ib_cast compilation units.
# deps/ must come before the base dir so stub headers (e.g. deps/gdaki/gin_host_gdaki.h)
# shadow CUDA-dependent originals in gdaki/.
set(NET_IB_CAST_INCLUDE_DIRS
    ${HIPIFY_DIR}/src/transport/net_ib_cast/deps
    ${HIPIFY_DIR}/src/transport/net_ib_cast
    ${HIPIFY_DIR}/src/transport/net_ib_cast/gdaki
    # Source-tree paths for headers not processed by hipify (no CUDA/HIP constructs)
    ${PROJECT_SOURCE_DIR}/src/transport/net_ib_cast/deps
    ${PROJECT_SOURCE_DIR}/src/transport/net_ib_cast
    ${PROJECT_SOURCE_DIR}/src/transport/net_ib_cast/gdaki
)

# Export to parent scope for consumption by src/CMakeLists.txt
set(NET_IB_CAST_SOURCES      ${NET_IB_CAST_SOURCES}      PARENT_SCOPE)
set(NET_IB_CAST_INCLUDE_DIRS ${NET_IB_CAST_INCLUDE_DIRS} PARENT_SCOPE)

# Add DOCA device headers to parent scope
set(DEVICE_DOCA_HEADERS ${DEVICE_DOCA_HEADERS} PARENT_SCOPE)

# Add DOCA sources to parent scope
set(DOCA_SOURCES ${DOCA_SOURCES} PARENT_SCOPE)

# Add DOCA_HOME to parent scope
set(DOCA_HOME ${DOCA_HOME} PARENT_SCOPE)
