#!/usr/bin/env python3
# generate triplet weights
from networkx.algorithms.structuralholes import constraint


def main_cli():
    import argparse
    import itertools
    import sys
    from typing import List

    import numpy as np
    from scipy.optimize import LinearConstraint, minimize

    from build_algo.util import read_phylip

    parser = argparse.ArgumentParser(
        prog="get_triplet_svd_weights", description="generate triplet weights"
    )
    parser.add_argument("--phylip", required=True, help="phylip file")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument(
        "--stationary_dist",
        type=float,
        nargs=4,
        default=[0.25, 0.25, 0.25, 0.25],
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=10,
    )

    if hasattr(sys, "ps1"):
        # default args for pasting into ipython
        opt = parser.parse_args(
            "--phylip data/alignment_GTR_30_taxa_1K_sites_rtree1.phy " "--output test.txt".split()
        )
    else:
        opt = parser.parse_args()
    print(opt)

    stationary_dist = np.array(opt.stationary_dist, dtype=np.float64)
    rank: int = opt.rank
    assert 0 < rank < 16, "invalid rank"

    seq_len, data = read_phylip(opt.phylip)

    with open(opt.output, "w") as file:
        for a_species, b_species, c_species in itertools.combinations(data.keys(), 3):

            P = make_count_tensor(data[a_species], data[b_species], data[c_species], seq_len)
            svd_score = np.zeros(3, dtype=np.float64)

            # fake quartet
            P_enh = np.einsum("ijk,l->ijkl", P, stationary_dist)
            split_patterns = [(0, 1, 2, 3), (0, 2, 1, 3), (1, 2, 0, 3)]
            for cherry_idx, transpose_pattern in enumerate(split_patterns):
                P_cherry_flat = P_enh.transpose(transpose_pattern).reshape(16, 16)

                svs: np.ndarray = np.linalg.svd(P_cherry_flat[:, :], compute_uv=False)
                # noinspection PyTypeChecker
                svd_score[cherry_idx] = np.sqrt(np.mean(svs[rank:] ** 2) / np.mean(svs[:rank] ** 2))

                p_hat = P_cherry_flat / np.sum(P_cherry_flat)
                U, S, Vh = np.linalg.svd(p_hat, compute_uv=True, full_matrices=True)
                low_rank_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]

                # fix the LRA up
                # sum to 1
                low_rank_approx += (1 - np.sum(low_rank_approx)) / 256

                # projector onto subspace defined by the LRA
                proj = np.identity(16) - (Vh[rank:, :].T @ Vh[rank:, :])

                # fix any negatives while keeping inside the LRA defined space
                min_loc = np.unravel_index(np.argmin(low_rank_approx), low_rank_approx.shape)
                fix_count = 0
                while low_rank_approx[min_loc] < 0.0 and fix_count <= 256:
                    compensation = np.zeros((16, 16))
                    compensation[min_loc] = 1.0
                    compensation -= 1 / 256
                    compensation = compensation @ proj
                    if compensation[min_loc] != 0.0:
                        low_rank_approx -= (
                            low_rank_approx[min_loc] / compensation[min_loc]
                        ) * compensation
                        fix_count += 1
                    else:
                        break

                    # sum to 1
                    low_rank_approx += (1 - np.sum(low_rank_approx)) / 256

                    min_loc = np.unravel_index(np.argmin(low_rank_approx), low_rank_approx.shape)

                test_statistic = np.sum(P_cherry_flat) * np.sum(
                    (p_hat - low_rank_approx) ** 2 / low_rank_approx
                )

                # constraint = LinearConstraint(
                #     A=np.concatenate(
                #         ([np.ones(256)], np.identity(256), np.kron(np.identity(16), Vh[rank:, :])),
                #         axis=0,
                #     ),
                #     lb=np.concatenate(
                #         [
                #             [1.0],
                #             [0.0] * 256,
                #             [0.0] * (16 - rank) * 16,
                #         ]
                #     ),
                #     ub=np.concatenate(
                #         [
                #             [1.0],
                #             [1.0] * 256,
                #             [0.0] * (16 - rank) * 16,
                #         ]
                #     ),
                # )
                #
                # def objective(x):
                #     return np.linalg.norm(p_hat.reshape(-1) - x)
                #
                # res = minimize(
                #     objective,
                #     init_low_rank_approx.reshape(-1),
                #     method="COBYLA",
                #     constraints=constraint,
                # )

                print(np.linalg.norm(P_cherry_flat - low_rank_approx) * 40 / np.mean(S[:4]))
                print(svs.astype(np.float16))
                print()

            # # triplet
            # split_patterns = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
            # for cherry_idx, transpose_pattern in enumerate(split_patterns):
            #     P_cherry_flat = P.transpose(transpose_pattern).reshape(16, 4)
            #     svs = np.linalg.svd(P_cherry_flat[:, :], compute_uv=False)
            #     print(svs.astype(np.float16))
            #     print()
            #     # noinspection PyTypeChecker
            #     svd_score[cherry_idx] = np.sqrt(np.mean(svs[rank:] ** 2) / np.mean(svs[:rank] ** 2))

            min_score_loc = np.argmin(svd_score)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_scores = np.nan_to_num(
                    np.float64(1.0) / svd_score, nan=0.0, posinf=100.0, neginf=0.0
                )
            weight = inv_scores[min_score_loc] / np.sum(inv_scores)

            triplet_species: List[str] = [a_species, b_species, c_species]
            file.write(
                f"{triplet_species[split_patterns[min_score_loc][0]]}"
                f" {triplet_species[split_patterns[min_score_loc][1]]}"
                f" {triplet_species[split_patterns[min_score_loc][2]]}"
                f" {weight}\n"
            )


def make_count_tensor(a_seq, b_seq, c_seq, seq_len):
    from functools import reduce
    from itertools import product
    from operator import mul

    import numpy as np

    P = np.zeros((4, 4, 4), dtype=np.float64)
    gaps = 0  # TODO: use this or lose this? counts number of sites that have a gap in any of the sequences
    for site_idx in range(seq_len):
        count = reduce(mul, map(lambda x: int(x[site_idx]).bit_count(), [a_seq, b_seq, c_seq]))

        if count == 0:
            gaps += 1
        else:
            # find the non-zero bits of the sequence and fill in the array
            a_opts = [k for k in range(4) if a_seq[site_idx] & (1 << k)]
            b_opts = [k for k in range(4) if b_seq[site_idx] & (1 << k)]
            c_opts = [k for k in range(4) if c_seq[site_idx] & (1 << k)]
            for a_opt, b_opt, c_opt in product(a_opts, b_opts, c_opts):
                P[a_opt, b_opt, c_opt] += 1 / count
    return P


if __name__ == "__main__":
    main_cli()
