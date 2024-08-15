import itertools

import numpy as np
import scipy


# noinspection PyPep8Naming
def cofactor(M):
    # noinspection PyShadowingNames
    N = np.zeros_like(M)
    for r, c in itertools.product(range(M.shape[0]), range(M.shape[1])):
        # noinspection PyTypeChecker
        N[r, c] = np.linalg.det(
            np.block(
                [[M[0:r, 0:c], M[0:r, c + 1:]], [M[r + 1:, 0:c], M[r + 1:, c + 1:]]]
            )
        )
    return N


# noinspection PyPep8Naming
def GTR(pis_, rcs_):
    pi_a, pi_c, pi_g, pi_t = pis_
    a, b, c, d, e, f = rcs_
    Q_ = np.array(
        [
            [-(a * pi_c + b * pi_g + c * pi_t), a * pi_c, b * pi_g, c * pi_t],
            [a * pi_c, -(a * pi_c + d * pi_g + e * pi_t), d * pi_g, e * pi_t],
            [b * pi_a, d * pi_c, -(b * pi_a + d * pi_c + f * pi_t), f * pi_t],
            [c * pi_a, e * pi_c, f * pi_g, -(c * pi_a + e * pi_c + f * pi_g)],
        ]
    )
    return Q_


pis = (0.3, 0.25, 0.2, 0.25)
rcs = (1, 1, 1, 1, 1, 1)

N = 1000

Q = GTR(pis, rcs)


def sample(pis_):
    return int(np.argmin(np.cumsum(pis_) < np.random.rand()))


# noinspection PyPep8Naming,PyShadowingNames
def cond_sample(P, D):
    return [sample(P[d, :]) for d in D]


def tree(ai, bi, i, ic):
    root_dist = [sample(pis) for _ in range(N)]
    i_dist = cond_sample(scipy.linalg.expm(i * Q), root_dist)
    a_dist = cond_sample(scipy.linalg.expm(ai * Q), i_dist)
    b_dist = cond_sample(scipy.linalg.expm(bi * Q), i_dist)
    c_dist = cond_sample(scipy.linalg.expm(ic * Q), root_dist)

    return np.array(a_dist), np.array(b_dist), np.array(c_dist)


a_seq, b_seq, c_seq = tree(1, 1, 1, 2)
P_sim = np.zeros((4, 4, 4), dtype=np.float64)
for site_idx in range(N):
    P_sim[a_seq[site_idx], b_seq[site_idx], c_seq[site_idx]] += 1

P_sim_false1 = P_sim.transpose((0, 2, 1))
P_sim_false2 = P_sim.transpose((1, 2, 0))

for P in [P_sim, P_sim_false1, P_sim_false2]:
    for k in range(4):
        A = P[:, :, k]
        B = np.sum(P, axis=1)
        C = np.sum(P, axis=0)
        D = -A.T
        E = -B.T
        F = -C.T
        print(
            np.linalg.svd(
                C @ np.linalg.inv(B) @ A + D @ np.linalg.inv(E) @ F, compute_uv=False
            )
        )
    print()

# for a_species, b_species, c_species in itertools.combinations(data.keys(), 3):
#     P_ab = np.zeros((4, 4, 4), dtype=np.float64)
#     a_seq, b_seq, c_seq = data[a_species], data[b_species], data[c_species]
#     for site_idx in range(seq_len):
#         P_ab[a_seq[site_idx], b_seq[site_idx], c_seq[site_idx]] += 1
#
#     for k in range(4):
#         A = P_ab[:, :, k]
#         B = np.sum(P_ab, axis=1)
#         C = np.sum(P_ab, axis=0)
#         D = -A.T
#         E = -B.T
#         F = -C.T
#         print(
#             np.linalg.svd(
#                 C @ np.linalg.inv(B) @ A + D @ np.linalg.inv(E) @ F, compute_uv=False
#             )
#         )
