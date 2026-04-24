/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stub for compiler.h — provides compiler portability macros.
 * The original is in the NCCL source tree but not present in RCCL.
 *************************************************************************/

#ifndef NCCL_COMPILER_H_
#define NCCL_COMPILER_H_

/* Branch prediction hints */
#ifndef NCCL_LIKELY
#define NCCL_LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef NCCL_UNLIKELY
#define NCCL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

/* Inline / noinline hints */
#ifndef NOINLINE
#define NOINLINE __attribute__((noinline))
#endif
#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE inline __attribute__((always_inline))
#endif

#endif /* NCCL_COMPILER_H_ */
