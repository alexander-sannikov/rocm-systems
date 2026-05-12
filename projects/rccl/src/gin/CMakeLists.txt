# Gin sources
set(GIN_SOURCES
    src/gin/gin_host.cc
    src/gin/gin_host_proxy.cc
)

# Add gin sources to parent scope
set(GIN_SOURCES ${GIN_SOURCES} PARENT_SCOPE)
