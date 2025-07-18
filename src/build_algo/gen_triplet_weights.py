#!/usr/bin/env python3
# generate triplet weights

import argparse
import itertools
import sys
from typing import Dict, List

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--phylip", required=True, help="phylip file")
parser.add_argument("--output", required=True, help="output file")
parser.add_argument(
    "--method",
    choices=["J", "P"],
    required=True,
    help="use Julia's (J) or the Paper's (P) array structure",
)

if hasattr(sys, "ps1"):
    # default args for pasting into ipython
    opt = parser.parse_args(
        "--phylip data/alignment_GTR_30_taxa_1K_sites_rtree1.phy "
        "--output test.txt "
        "--method J".split()
    )
else:
    opt = parser.parse_args()
print(opt)


def seq_to_array(seq: str) -> np.ndarray:
    arr = np.zeros(len(seq), dtype=np.int8)
    a = 0b0001
    c = 0b0010
    g = 0b0100
    t = 0b1000
    for idx_, c_ in enumerate(seq):
        match c_.upper():
            case "A":
                arr[idx_] = a
            case "C":
                arr[idx_] = c
            case "G":
                arr[idx_] = g
            case "T":
                arr[idx_] = t
            case "R":
                arr[idx_] = a + g
            case "Y":
                arr[idx_] = c + t
            case "S":
                arr[idx_] = g + c
            case "W":
                arr[idx_] = a + t
            case "K":
                arr[idx_] = g + t
            case "M":
                arr[idx_] = a + c
            case "B":
                arr[idx_] = c + g + t
            case "D":
                arr[idx_] = a + g + t
            case "H":
                arr[idx_] = a + c + t
            case "V":
                arr[idx_] = a + c + g
            case "N":
                arr[idx_] = a + c + g + t
            case _:
                # gap
                arr[idx_] = 0
    return arr


data: Dict[str, np.ndarray] = dict()
with open(opt.phylip, "r") as file:
    n_species, seq_len = map(lambda s: int(s.strip()), file.readline().strip().split())

    for line in file:
        species, *sequence = line.strip().split()
        sequence = "".join(sequence)
        assert len(sequence) == seq_len, f"seq length mismatch for {species}"
        data[species] = seq_to_array(sequence)

with open(opt.output, "w") as file:
    for a_species, b_species, c_species in itertools.combinations(data.keys(), 3):
        P = np.zeros((4, 4, 4), dtype=np.float64)
        a_seq, b_seq, c_seq = data[a_species], data[b_species], data[c_species]
        gaps = 0  # TODO: use this or lose this? counts number of sites that have a gap in any of the sequences
        for site_idx in range(seq_len):
            count = (
                int(a_seq[site_idx]).bit_count()
                * int(b_seq[site_idx]).bit_count()
                * int(c_seq[site_idx]).bit_count()
            )
            if count == 0:
                gaps += 1
            else:
                # find the non-zero bits of the sequence and fill in the array
                a_opts = [k for k in range(4) if a_seq[site_idx] & (1 << k)]
                b_opts = [k for k in range(4) if b_seq[site_idx] & (1 << k)]
                c_opts = [k for k in range(4) if c_seq[site_idx] & (1 << k)]
                for a_opt, b_opt, c_opt in itertools.product(a_opts, b_opts, c_opts):
                    P[a_opt, b_opt, c_opt] += 1 / count

        triplet_patterns = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        svd_score = np.zeros(3, dtype=np.float64)
        for cherry_idx, transpose_pattern in enumerate(triplet_patterns):
            score = 0.0
            P_cherry = P.transpose(transpose_pattern)
            for k in range(4):
                M = np.zeros((12, 12), dtype=np.float64)
                if opt.method == "J":
                    # Julia:
                    M[0:4, 4:8] = np.sum(P_cherry, axis=0).T
                    M[0:4, 8:12] = np.sum(P_cherry, axis=1).T
                    M[4:8, 0:4] = -np.sum(P_cherry, axis=0)
                    M[4:8, 8:12] = P_cherry[:, :, k]
                    M[8:12, 0:4] = -np.sum(P_cherry, axis=1)
                    M[8:12, 4:8] = -P_cherry[:, :, k].T
                elif opt.method == "P":
                    # paper:
                    M[0:4, 4:8] = np.sum(P_cherry, axis=1).T
                    M[0:4, 8:12] = np.sum(P_cherry, axis=0).T
                    M[4:8, 0:4] = -np.sum(P_cherry, axis=1)
                    M[4:8, 8:12] = P_cherry[:, :, k]
                    M[8:12, 0:4] = -np.sum(P_cherry, axis=0)
                    M[8:12, 4:8] = -P_cherry[:, :, k].T
                else:
                    assert False

                svs = np.linalg.svd(M[:, :], compute_uv=False).astype(np.float32).astype(np.float64)
                score += np.mean(svs[8:] ** 2) / np.mean(svs[:8] ** 2)

            for k in range(4):
                N = np.zeros((12, 12), dtype=np.float64)
                if opt.method == "J":
                    # Julia:
                    N[0:4, 4:8] = P_cherry[:, :, k]
                    N[0:4, 8:12] = np.sum(P_cherry, axis=0)
                    N[4:8, 0:4] = -P_cherry[:, :, k].T
                    N[4:8, 8:12] = np.sum(P_cherry, axis=1)
                    N[8:12, 0:4] = -np.sum(P_cherry, axis=0).T
                    N[8:12, 4:8] = -np.sum(P_cherry, axis=1).T
                elif opt.method == "P":
                    # paper:
                    N[0:4, 4:8] = P_cherry[:, :, k]
                    N[0:4, 8:12] = np.sum(P_cherry, axis=1)
                    N[4:8, 0:4] = -P_cherry[:, :, k].T
                    N[4:8, 8:12] = np.sum(P_cherry, axis=0)
                    N[8:12, 0:4] = -np.sum(P_cherry, axis=1).T
                    N[8:12, 4:8] = -np.sum(P_cherry, axis=0).T
                else:
                    assert False

                svs = np.linalg.svd(N[:, :], compute_uv=False).astype(np.float32).astype(np.float64)
                score += np.mean(svs[8:] ** 2) / np.mean(svs[:8] ** 2)

            svd_score[cherry_idx] = np.sqrt(score)

        min_score_loc = np.argmin(svd_score)
        inv_scores = np.nan_to_num(1 / svd_score)
        weight = inv_scores[min_score_loc] / np.sum(inv_scores)

        triplet_species: List[str] = [a_species, b_species, c_species]
        file.write(
            f"{triplet_species[triplet_patterns[min_score_loc][0]]}"
            f" {triplet_species[triplet_patterns[min_score_loc][1]]}"
            f" {triplet_species[triplet_patterns[min_score_loc][2]]}"
            f" {weight}\n"
        )
