/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mlx5/mlx5dvwrap.h"
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#include <dlfcn.h>

#ifdef NCCL_BUILD_MLX5DV
#include <infiniband/mlx5dv.h>
#else
#include "mlx5/mlx5dvcore.h"
#endif
#include "mlx5/mlx5dvsymbols.h"

static std::once_flag initOnceFlag;
static ncclResult_t initResult;
struct ncclMlx5dvSymbols mlx5dvSymbols;

ncclResult_t wrap_mlx5dv_symbols(void) {
  std::call_once(initOnceFlag,
               [](){ initResult = buildMlx5dvSymbols(&mlx5dvSymbols); });
  return initResult;
}

/* CHECK_NOT_NULL: helper macro to check for NULL symbol */
#define CHECK_NOT_NULL(container, internal_name) \
  if (container.internal_name == NULL) { \
     WARN("NET/MLX5: lib wrapper not initialized."); \
     return ncclInternalError; \
  }

#define MLX5DV_PTR_CHECK_ERRNO(container, internal_name, call, retval, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  retval = container.call; \
  if (retval == error_retval) { \
    WARN("NET/MLX5: Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

bool wrap_mlx5dv_is_supported(struct ibv_device *device) {
  if (mlx5dvSymbols.mlx5dv_internal_is_supported == NULL) {
    return 0;
  }
  return mlx5dvSymbols.mlx5dv_internal_is_supported(device);
}

ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context* context, char* buf, size_t buf_len) {
  CHECK_NOT_NULL(mlx5dvSymbols, mlx5dv_internal_get_data_direct_sysfs_path);
  int ret = mlx5dvSymbols.mlx5dv_internal_get_data_direct_sysfs_path(context, buf, buf_len);
  if (ret == 0) return ncclSuccess;
  /* ENODEV can happen if the devices is not data-direct but mlx5 is used. It's not an error*/
  if (ret == ENODEV) return ncclInvalidArgument;
  INFO(NCCL_NET, "NET/MLX5: Call to mlx5dv_internal_get_data_direct_sysfs_path failed with error %s errno %d", strerror(ret), ret);
  return ncclSystemError;
}

/* DMA-BUF support */
ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access) {
  MLX5DV_PTR_CHECK_ERRNO(mlx5dvSymbols, mlx5dv_internal_reg_dmabuf_mr, mlx5dv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access), *ret, NULL, "mlx5dv_reg_dmabuf_mr");
}

struct ibv_mr * wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access) {
  if (mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr == NULL) {
    errno = EOPNOTSUPP; // ncclIbDmaBufSupport() requires this errno being set
    return NULL;
  }
  return mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access);
}

/* OOO RQ capability query — dynamically resolved at runtime. Returns ncclInternalError if unavailable. */
ncclResult_t wrap_mlx5dv_query_device(struct ibv_context *context, struct mlx5dv_context *attrs_out) {
#ifdef NCCL_BUILD_MLX5DV
  int ret = mlx5dv_query_device(context, attrs_out);
  if (ret != 0) return ncclInternalError;
  return ncclSuccess;
#else
  typedef int (*mlx5dv_query_device_fn)(struct ibv_context*, struct mlx5dv_context*);
  static mlx5dv_query_device_fn fn = NULL;
  static bool resolved = false;
  if (!resolved) {
    fn = (mlx5dv_query_device_fn)dlsym(RTLD_DEFAULT, "mlx5dv_query_device");
    resolved = true;
  }
  if (fn == NULL) return ncclInternalError;
  int ret = fn(context, attrs_out);
  if (ret != 0) return ncclInternalError;
  return ncclSuccess;
#endif
}

/* OOO QP creation — AINIC path. Dynamically resolved at runtime; returns NULL if unavailable. */
struct ibv_qp * wrap_mlx5dv_create_qp(struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_attr, struct mlx5dv_qp_init_attr *mlx5_qp_attr) {
#ifdef NCCL_BUILD_MLX5DV
  return mlx5dv_create_qp(context, qp_attr, mlx5_qp_attr);
#else
  /* Dynamic lookup via dlsym — fall back to NULL if mlx5dv_create_qp not available */
  typedef struct ibv_qp* (*mlx5dv_create_qp_fn)(struct ibv_context*, struct ibv_qp_init_attr_ex*, struct mlx5dv_qp_init_attr*);
  static mlx5dv_create_qp_fn fn = NULL;
  static bool resolved = false;
  if (!resolved) {
    fn = (mlx5dv_create_qp_fn)dlsym(RTLD_DEFAULT, "mlx5dv_create_qp");
    resolved = true;
  }
  if (fn == NULL) { errno = EOPNOTSUPP; return NULL; }
  return fn(context, qp_attr, mlx5_qp_attr);
#endif
}
