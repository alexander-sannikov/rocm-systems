// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT 

#ifndef NCCL_MLX5DV_CORE_H_
#define NCCL_MLX5DV_CORE_H_

/* Basic MLX5 direct verbs structs. Needed to dynamically load MLX5 direct verbs functions without
 * explicit including of MLX5 direct verbs header.
 */

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include "ibvwrap.h"

enum mlx5dv_reg_dmabuf_access  {
	MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT		= (1<<0),
};

/* OOO (Out-of-Order) QP support — required by net_ib_cast AINIC path */
/* OOO Receive Queue capability query — used by IbCastGetOooRqSize() in init.cc */
enum mlx5dv_context_comp_mask {
  MLX5DV_CONTEXT_MASK_OOO_RECV_WRS = (1 << 6),
};

struct mlx5dv_ooo_recv_wrs_caps {
  uint32_t max_rc;
  uint32_t max_ud;
  uint32_t max_dc;
  uint32_t reserved[5];
};

struct mlx5dv_context {
  uint64_t comp_mask;
  struct mlx5dv_ooo_recv_wrs_caps ooo_recv_wrs_caps;
  uint8_t  reserved[128]; /* room for future fields */
};

enum mlx5dv_qp_create_flags {
	MLX5DV_QP_CREATE_OOO_DP = (1 << 0),
};

enum mlx5dv_qp_init_attr_mask {
	MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS = (1 << 0),
};

struct mlx5dv_qp_init_attr {
	uint64_t comp_mask;  /* Use mlx5dv_qp_init_attr_mask */
	uint32_t create_flags; /* Use mlx5dv_qp_create_flags */
	uint8_t  dc_init_attr[32]; /* reserved for DC QP attrs */
};

#endif  // NCCL_MLX5DV_CORE_H_
