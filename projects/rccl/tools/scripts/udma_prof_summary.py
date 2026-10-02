#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Merge per-rank RCCL UDMA profiler JSON dumps into two tables:
#   1) QP counts by nic / udma / type / dest
#   2) Posted bytes by nic / udma / type
#
# Usage:
#   python3 tools/scripts/udma_prof_summary.py /tmp/udma/rccl_udma_prof.*.json

import argparse
import glob
import json
import sys
from collections import defaultdict


def load_files(paths):
    docs = []
    for path in paths:
        with open(path, encoding="utf-8") as fp:
            docs.append(json.load(fp))
    return docs


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(description="Summarize RCCL UDMA profiler JSON dumps")
    parser.add_argument("files", nargs="+", help="rccl_udma_prof.<host>.<pid>.json files")
    args = parser.parse_args()

    paths = []
    for pattern in args.files:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    if not paths:
        print("no input files", file=sys.stderr)
        return 1

    docs = load_files(paths)

    qp_counts = defaultdict(int)
    bytes_tot = defaultdict(int)
    for doc in docs:
        src = doc.get("source", "?")
        for row in doc.get("qp_counts", []):
            key = (src, row.get("nic", ""), row.get("udma", ""), row.get("type", ""),
                   row.get("dst", ""))
            qp_counts[key] += int(row.get("count", 0))
        for row in doc.get("bytes", []):
            key = (src, row.get("nic", ""), row.get("udma", ""), row.get("type", ""))
            bytes_tot[key] += int(row.get("bytes", 0))

    print("QP assignment counts")
    print_table(
        ["source", "nic", "udma", "type", "dst", "qps"],
        [(*k, v) for k, v in sorted(qp_counts.items())],
    )
    print()
    print("Posted bytes")
    print_table(
        ["source", "nic", "udma", "type", "bytes"],
        [(*k, v) for k, v in sorted(bytes_tot.items())],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
