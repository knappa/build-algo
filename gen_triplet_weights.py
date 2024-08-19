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
    a = 0b0001
    c = 0b0010
    g = 0b0100
    t = 0b1000
    for idx, c_ in enumerate(seq):
        match c_.upper():
            case "A":
                arr[idx] = a
            case "C":
                arr[idx] = c
            case "G":
                arr[idx] = g
            case "T":
                arr[idx] = t
            case "R":
                arr[idx] = a + g
            case "Y":
                arr[idx] = c + t
            case "S":
                arr[idx] = g + c
            case "W":
                arr[idx] = a + t
            case "K":
                arr[idx] = g + t
            case "M":
                arr[idx] = a + c
            case "B":
                arr[idx] = c + g + t
            case "D":
                arr[idx] = a + g + t
            case "H":
                arr[idx] = a + c + t
            case "V":
                arr[idx] = a + c + g
            case "N":
                arr[idx] = a + c + g + t
            case _:
                # gap
                arr[idx] = 0
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
                    P_ab[a_opt, b_opt, c_opt] += 1 / count

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
                np.linalg.svd(M_ab[k, :, :], compute_uv=False)[10:] ** 2
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
                np.linalg.svd(M_ac[k, :, :], compute_uv=False)[10:] ** 2
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
                np.linalg.svd(M_bc[k, :, :], compute_uv=False)[10:] ** 2
            )
        svd_score_bc = np.sqrt(svd_score_bc)

        print(svd_score_ab, svd_score_bc, svd_score_bc)

        min_score = min([svd_score_ab, svd_score_ac, svd_score_bc])
        if svd_score_ab == min_score:
            file.write(f"{a_species} {b_species} {c_species} {svd_score_ab}\n")
        elif svd_score_ac == min_score:
            file.write(f"{a_species} {c_species} {b_species} {svd_score_ac}\n")
        elif svd_score_bc == min_score:
            file.write(f"{b_species} {c_species} {a_species} {svd_score_bc}\n")
