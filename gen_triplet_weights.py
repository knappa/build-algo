#!/usr/bin/env python3
# generate triplet weights

import argparse
import itertools
from typing import Dict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--phylip", required=True, help="phylip file")
parser.add_argument("--output", required=True, help="output file")


try:
    opt = parser.parse_args()
    print(opt)
except SystemExit:
    # options for pasting into ipython
    class Object:
        pass

    opt = Object()
    opt.phylip = "data/alignment_GTR_30_taxa_1K_sites_rtree1.phy"
    opt.output = "test.txt"


def seq_to_array(seq: str, seq_len_: int) -> np.ndarray:
    arr = np.zeros(seq_len_, dtype=np.int_)
    for idx, c_ in enumerate(seq):
        if c_ in {"A", "a"}:
            arr[idx] = 0
        elif c_ in {"C", "c"}:
            arr[idx] = 1
        elif c_ in {"G", "g"}:
            arr[idx] = 2
        elif c_ in {"T", "t"}:
            arr[idx] = 3
    return arr


data: Dict[str, np.ndarray] = dict()
with open(opt.phylip, "r") as file:
    n_species, seq_len = map(lambda s: int(s.strip()), file.readline().strip().split())

    for line in file:
        species, *sequence = line.strip().split()
        sequence = "".join(sequence)
        assert len(sequence) == seq_len, f"seq length mismatch for {species}"
        data[species] = seq_to_array(sequence, seq_len)


with open(opt.output, "w") as file:
    for a_species, b_species, c_species in itertools.combinations(data.keys(), 3):
        P_ab = np.zeros((4, 4, 4), dtype=np.float64)
        a_seq, b_seq, c_seq = data[a_species], data[b_species], data[c_species]
        for site_idx in range(seq_len):
            P_ab[a_seq[site_idx], b_seq[site_idx], c_seq[site_idx]] += 1

        svd_score_ab = 0.0
        M_ab = np.zeros((4, 12, 12), dtype=np.float64)
        for k in range(4):
            M_ab[k, 0:4, 4:8] = np.sum(P_ab, axis=1).T
            M_ab[k, 0:4, 8:12] = np.sum(P_ab, axis=0).T
            M_ab[k, 4:8, 0:4] = -np.sum(P_ab, axis=1)
            M_ab[k, 4:8, 8:12] = P_ab[:, :, k]
            M_ab[k, 8:12, 0:4] = -np.sum(P_ab, axis=0)
            M_ab[k, 8:12, 4:8] = -P_ab[:, :, k].T

            svd_score_ab += np.sum(
                np.linalg.svd(M_ab[k, :, :], compute_uv=False)[9:] ** 2
            )
        svd_score_ab = np.sqrt(svd_score_ab)

        P_ac = np.transpose(P_ab, axes=(0, 2, 1))
        svd_score_ac = 0.0
        M_ac = np.zeros((4, 12, 12), dtype=np.float64)
        for k in range(4):
            M_ac[k, 0:4, 4:8] = np.sum(P_ac, axis=1).T
            M_ac[k, 0:4, 8:12] = np.sum(P_ac, axis=0).T
            M_ac[k, 4:8, 0:4] = -np.sum(P_ac, axis=1)
            M_ac[k, 4:8, 8:12] = P_ac[:, :, k]
            M_ac[k, 8:12, 0:4] = -np.sum(P_ac, axis=0)
            M_ac[k, 8:12, 4:8] = -P_ac[:, :, k].T

            svd_score_ac += np.sum(
                np.linalg.svd(M_ac[k, :, :], compute_uv=False)[9:] ** 2
            )
        svd_score_ac = np.sqrt(svd_score_ac)

        P_bc = np.transpose(P_ab, axes=(1, 2, 0))
        svd_score_bc = 0.0
        M_bc = np.zeros((4, 12, 12), dtype=np.float64)
        for k in range(4):
            M_bc[k, 0:4, 4:8] = np.sum(P_bc, axis=1).T
            M_bc[k, 0:4, 8:12] = np.sum(P_bc, axis=0).T
            M_bc[k, 4:8, 0:4] = -np.sum(P_bc, axis=1)
            M_bc[k, 4:8, 8:12] = P_bc[:, :, k]
            M_bc[k, 8:12, 0:4] = -np.sum(P_bc, axis=0)
            M_bc[k, 8:12, 4:8] = -P_bc[:, :, k].T

            svd_score_bc += np.sum(
                np.linalg.svd(M_bc[k, :, :], compute_uv=False)[9:] ** 2
            )
        svd_score_bc = np.sqrt(svd_score_bc)

        min_score = min([svd_score_ab, svd_score_ac, svd_score_bc])
        if svd_score_ab == min_score:
            file.write(f"{a_species} {b_species} {c_species} {svd_score_ab}\n")
        elif svd_score_ac == min_score:
            file.write(f"{a_species} {c_species} {b_species} {svd_score_ac}\n")
        elif svd_score_bc == min_score:
            file.write(f"{b_species} {c_species} {a_species} {svd_score_bc}\n")
