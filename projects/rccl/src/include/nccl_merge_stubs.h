#ifndef NCCL_MERGE_STUBS_H_
#define NCCL_MERGE_STUBS_H_


// Temporary merge workaround: compiler.h and compiler/gcc.h are NCCL-only headers
// not present in RCCL. Inline the GCC/Clang compiler abstraction macros here.
#ifdef __cplusplus
extern "C++" {
#endif
#include <atomic>
#ifdef __cplusplus
}
#endif

#define NCCL_CONVERT_ORDER(order) \
  ((order) == std::memory_order_relaxed ? __ATOMIC_RELAXED : \
   (order) == std::memory_order_consume ? __ATOMIC_CONSUME : \
   (order) == std::memory_order_acquire ? __ATOMIC_ACQUIRE : \
   (order) == std::memory_order_release ? __ATOMIC_RELEASE : \
   (order) == std::memory_order_acq_rel ? __ATOMIC_ACQ_REL : \
   (order) == std::memory_order_seq_cst ? __ATOMIC_SEQ_CST : \
   __ATOMIC_SEQ_CST)

#define COMPILER_ATOMIC_LOAD(ptr, order) \
  __atomic_load_n((ptr), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) do { \
  __atomic_load((ptr), (dest), NCCL_CONVERT_ORDER(order)); \
} while(0)
#define COMPILER_ATOMIC_STORE(ptr, val, order) \
  __atomic_store_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_LOAD_32(ptr, order) \
  __atomic_load_n((ptr), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_STORE_32(ptr, val, order) \
  __atomic_store_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_EXCHANGE(ptr, val, order) \
  __atomic_exchange_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired, success_order, failure_order) \
  __atomic_compare_exchange_n((ptr), (expected), (desired), true, NCCL_CONVERT_ORDER(success_order), NCCL_CONVERT_ORDER(failure_order))

#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) __atomic_fetch_add((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_ADD_FETCH(ptr, val, order) __atomic_add_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_SUB_FETCH(ptr, val, order) __atomic_sub_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))

#define COMPILER_PREFETCH(addr)              __builtin_prefetch((addr))
#define COMPILER_POPCOUNT32(x)               __builtin_popcount(x)
#define COMPILER_POPCOUNT64(x)               __builtin_popcountll(x)
#define COMPILER_EXPECT(x, v)                __builtin_expect((x), (v))
#define COMPILER_FFS(x)                      __builtin_ffs(x)
#define COMPILER_FFSL(x)                     __builtin_ffsl(x)
#define COMPILER_FFSLL(x)                    __builtin_ffsll(x)
#define COMPILER_CLZ(x)                      __builtin_clz(x)
#define COMPILER_CLZL(x)                     __builtin_clzl(x)
#define COMPILER_CLZLL(x)                    __builtin_clzll(x)
#define COMPILER_BSWAP16(x)                  __builtin_bswap16(x)
#define COMPILER_BSWAP32(x)                  __builtin_bswap32(x)
#define COMPILER_BSWAP64(x)                  __builtin_bswap64(x)
#define COMPILER_ASSUME_ALIGNED(ptr, align)  __builtin_assume_aligned((ptr), (align))
#define COMPILER_ATTRIBUTE_UNUSED            __attribute__((unused))

// RCCL: NCCL_WIN_STRICT_ORDERING is in NCCL 2.30.4 nccl.h but not in RCCL nccl.h
#ifndef NCCL_WIN_STRICT_ORDERING
#define NCCL_WIN_STRICT_ORDERING 0x02
#endif

// Temporary merge workaround: os.h is NCCL-only; stub ncclOsGetPageSize here
#include <unistd.h>
static inline size_t ncclOsGetPageSize() { return (size_t)sysconf(_SC_PAGESIZE); }

// RCCL: ncclTopoGetLocalGinDevs stub — GIN topo not implemented in RCCL;
// returns device 0 with count 1 so gin_host.cc can proceed.
// Full topo-aware implementation would require ncclTopoGetLocalGinDev in topo.cc.
#ifndef _NCCL_TOPO_GIN_DEVS_STUB_
#define _NCCL_TOPO_GIN_DEVS_STUB_
struct ncclComm;
static inline ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* /*comm*/, int* localGinDevs, int* localGinCount) {
  localGinDevs[0] = 0;  // default to first IB device
  *localGinCount = 1;
  return ncclSuccess;
}
#endif

// RCCL: ncclSetThreadName overload for std::thread (NCCL 2.30.4 uses std::thread,
// RCCL's existing signature takes pthread_t)
#ifdef __cplusplus
#include <thread>
#include <cstdio>
#include <cstdarg>
#ifndef _NCCL_SET_THREAD_NAME_STD_THREAD_
#define _NCCL_SET_THREAD_NAME_STD_THREAD_
// Forward decl of the pthread_t overload (defined in debug.cc)
void ncclSetThreadName(pthread_t thread, const char *fmt, ...);
static inline void ncclSetThreadName(std::thread& thread, const char *fmt, ...) {
  char name[16];
  va_list args;
  va_start(args, fmt);
  vsnprintf(name, sizeof(name), fmt, args);
  va_end(args);
  ncclSetThreadName(thread.native_handle(), "%s", name);
}
#endif
#endif // __cplusplus

#endif // NCCL_MERGE_STUBS_H_
