# Gin plugin sources
set(PLUGIN_GIN_SOURCES
    src/plugin/gin/gin_v11.cc
    src/plugin/gin/gin_v12.cc
    src/plugin/gin/gin_v13.cc
)

# Add net plugin sources to parent scope
set(PLUGIN_GIN_SOURCES ${PLUGIN_GIN_SOURCES} PARENT_SCOPE)
