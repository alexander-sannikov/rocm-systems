/*************************************************************************
 * RCCL stub for NCCL 2.30.4 os.h — Linux/ROCm only minimal shim
 * Provides ncclOsDlsym used by gin_v11.cc and gin_v12.cc
 *************************************************************************/

#ifndef NCCL_OS_H_
#define NCCL_OS_H_

#include <dlfcn.h>
#include <unistd.h>
#include <cstdint>
#include <cstddef>

// RCCL: os.h stub — only the symbols needed by plugin/gin/gin_v1{1,2}.cc
typedef void* ncclOsLibraryHandle;

static inline ncclOsLibraryHandle ncclOsDlopen(const char* filename) {
  return dlopen(filename, RTLD_NOW | RTLD_LOCAL);
}

static inline void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) {
  return dlsym(handle, symbol);
}

static inline void ncclOsDlclose(ncclOsLibraryHandle handle) {
  dlclose(handle);
}

static inline const char* ncclOsDlerror() {
  return dlerror();
}

static inline size_t ncclOsGetPageSize() {
  return (size_t)sysconf(_SC_PAGESIZE);
}

#endif // NCCL_OS_H_
