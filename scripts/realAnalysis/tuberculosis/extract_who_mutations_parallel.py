#!/usr/bin/env python3
"""
Faster parallel WHO mutation extractor.

Requirements:
  pip install cyvcf2 pandas numpy

Usage:
  - Put this script in your project root (or assemblies/)
  - Ensure WHO file path WHO_FILE is correct (default ../who_mutations.csv if script in assemblies/)
  - Ensure VCF files are indexed (.csi or .tbi). You have bcftools index earlier.
  - Run: python extract_who_mutations_parallel.py
"""

import os
import glob
import numpy as np
import pandas as pd
from cyvcf2 import VCF
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ------------- CONFIG -------------
path = "/Users/sijianfan/Documents/projects/BiSSGL/datasets/realAnalysis/tuberculosis"
WHO_FILE = f"{path}/who_mutations.csv"  # WHO 突变定义文件
VCF_PATTERN = f"{path}/assemblies/*.raw.vcf.gz"  # 匹配你的 VCF 文件名
OUTPUT = f"{path}/mutation_matrix_who_sites.csv"
MIN_COV = 10  # 覆盖度阈值（低于该值视为不可调用 NaN）
MAX_WORKERS = min(5, max(1, multiprocessing.cpu_count() - 2))  # safe default
VERBOSE = True
# -----------------------------------


def load_who(who_fn):
    df = pd.read_csv(who_fn, dtype={"pos": int})
    required = {"mut_name", "pos", "ref", "alt"}
    if not required.issubset(df.columns):
        raise ValueError(f"who_mutations.csv must contain columns {required}")
    df = df.drop_duplicates(subset=["mut_name"])
    # group by chrom if you have multiple contigs; we assume CHROM = NC_000962.3 for MTB
    # create list of tuples for quick iteration
    who_rows = df.to_dict(orient="records")
    return df, who_rows


def process_one_vcf(vcf_path, who_rows, min_cov=MIN_COV):
    """
    Process a single vcf file by fetching only WHO positions.
    Returns (acc, {mut_name: value, ...})
    """
    acc = os.path.basename(vcf_path).replace(".raw.vcf.gz", "")
    # open vcf
    v = VCF(vcf_path)
    # prepare row
    row = {w["mut_name"]: np.nan for w in who_rows}
    # If WHO positions are all on single chrom, you can use that; else use w['chrom'] if you included it.
    # We assume CHROM in VCF is NC_000962.3 and WHO pos are for that chrom.
    for w in who_rows:
        pos = int(w["pos"])
        ref_w = str(w["ref"])
        alt_w = str(w["alt"])
        mut = w["mut_name"]
        # cyvcf2.fetch is 0-based half-open: fetch(chrom, start, end) -> start = pos-1, end = pos
        try:
            recs = list(v.fetch("NC_000962.3", pos - 1, pos))
        except Exception:
            # fallback: iterate entire vcf once (shouldn't happen if properly indexed); mark NaN to be safe
            row[mut] = np.nan
            continue

        if not recs:
            row[mut] = np.nan
            continue

        seen = False
        callable_flag = False
        for rec in recs:
            # get dp
            try:
                dp_arr = rec.format("DP")
                dp = int(dp_arr[0]) if dp_arr is not None else 0
            except Exception:
                dp = 0
            if dp < min_cov:
                continue
            callable_flag = True
            rec_ref = str(rec.REF)
            rec_alts = [str(a) for a in rec.ALT]
            if rec_ref == ref_w and alt_w in rec_alts:
                seen = True
        if not callable_flag:
            row[mut] = np.nan
        else:
            row[mut] = 1 if seen else 0
    # close vcf (cyvcf2 cleans up on GC but be explicit)
    v.close()
    return acc, row


def main():
    # find vcf files (script assumed running in assemblies/)
    vcf_files = sorted(glob.glob(VCF_PATTERN))
    if len(vcf_files) == 0:
        raise RuntimeError(f"No VCF files found matching {VCF_PATTERN}")

    who_df, who_rows = load_who(WHO_FILE)
    mut_names = who_df["mut_name"].tolist()
    if VERBOSE:
        print(f"WHO mutations: {len(mut_names)}")
        print(f"VCF files: {len(vcf_files)}")
        print(f"Using {MAX_WORKERS} workers")

    rows = []
    idx = []

    # parallel processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {
            exe.submit(process_one_vcf, vcf, who_rows, MIN_COV): vcf
            for vcf in vcf_files
        }
        for future in as_completed(futures):
            vcf = futures[future]
            try:
                acc, row = future.result()
                idx.append(acc)
                rows.append(row)
                if VERBOSE:
                    print(f"[DONE] {acc}")
            except Exception as e:
                print(f"[ERROR] {vcf} -> {e}")

    # assemble dataframe (preserve order of who mutations)
    mat = pd.DataFrame(rows, index=idx)
    mat.index.name = "assembly_acc"
    mat = mat.reindex(columns=mut_names)
    mat.to_csv(OUTPUT)
    print(f"Wrote {OUTPUT} shape={mat.shape}")


if __name__ == "__main__":
    main()
