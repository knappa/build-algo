#!/usr/bin/env python3
# implements a weighted BUILD algorithm based on Semple and Steel
import argparse
import itertools
import os.path
import subprocess
import tempfile
from functools import reduce
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy
from ete3 import Tree

parser = argparse.ArgumentParser(
    description="Use a version of the BUILD algorithm on triplets to do tree reconstruction"
)
parser.add_argument("--triplets", required=True, help="triplets file")
parser.add_argument(
    "--out",
    type=str,
    help="output file (Newick format). If unspecified, print to stdout",
)
parser.add_argument(
    "--method",
    choices=["P", "L", "MCL"],
    required=True,
    help="Markov (P), Laplacian (L), or MCL clustering method",
)

opt = parser.parse_args()
print(opt)


def np_full_print(nparray):
    import shutil

    # noinspection PyTypeChecker
    with np.printoptions(threshold=np.inf, linewidth=shutil.get_terminal_size((80, 20)).columns):
        print(nparray)


def get_data(trip_file, file_weights=True):
    species: Set[str] = set()
    triplets: List[Tuple[str, str, str]] = list()
    weights: Dict[Tuple[str, str, str], float] = dict()

    with open(trip_file, "r") as t_file:
        for line in t_file:
            a, b, c, *weight = line.strip().split(" ")
            triplets.append((a, b, c))
            species = species.union([a, b, c])
            if file_weights:
                weights[(a, b, c)] = float(weight[0])
            else:
                weights[(a, b, c)] = 1.0

    return species, triplets, weights


all_species, all_triplets, all_weights = get_data(opt.triplets)


def gen_tree_weighted(
    triplets: List[Tuple[str, str, str]],
    weights: Dict[Tuple[str, str, str], float],
    *,
    node=None,
    tree=None,
):
    if (node is not None and tree is None) or (node is None and tree is not None):
        assert "inconsistent state"
    if tree is None:
        tree = Tree()
        node = tree

    if len(triplets) == 1:
        a, b, c = triplets[0]
        # noinspection PyTypeChecker
        node.add_child(name=c)
        subnode = node.add_child()
        subnode.add_child(name=a)
        subnode.add_child(name=b)
        return tree

    species: list = list(set(reduce(set.union, triplets, set())))
    species.sort()
    species_to_index = {s: i for i, s in enumerate(species)}
    adj_count_matrix = np.zeros((len(species), len(species)), dtype=np.float64)
    adj_weight_matrix = np.zeros((len(species), len(species)), dtype=np.float64)
    for a, b, c in triplets:
        adj_count_matrix[species_to_index[a], species_to_index[b]] += 1
        adj_count_matrix[species_to_index[b], species_to_index[a]] += 1
        adj_weight_matrix[species_to_index[a], species_to_index[b]] += weights[(a, b, c)]
        adj_weight_matrix[species_to_index[b], species_to_index[a]] += weights[(a, b, c)]

    with np.errstate(divide="ignore", invalid="ignore"):
        adj_matrix = np.nan_to_num(adj_weight_matrix / adj_count_matrix, nan=0.0)

    match opt.method:
        case "P":
            components = markov_p(adj_matrix=adj_matrix, species=species)
        case "L":
            components = spectral_laplacian(adj_matrix=adj_matrix, species=species)
        case "MCL":
            components = markov_clustering(
                adj_matrix=adj_matrix,
                species=species,
                species_to_index=species_to_index,
            )
        case _:
            assert False

    assert len(components) > 1, "No splitting found"
    assert all(len(component) > 0 for component in components), "One split was empty"

    for comp in components:
        if len(comp) == 1:
            member = list(comp)[0]
            node.add_child(name=member)
        elif len(comp) > 1:
            # filter triplets by component
            comp_triplets = [
                (a, b, c) for (a, b, c) in triplets if all([x in comp for x in [a, b, c]])
            ]
            subnode = node.add_child()
            if len(comp_triplets) == 0:
                for member in comp:
                    subnode.add_child(name=member)
            else:
                gen_tree_weighted(comp_triplets, weights, node=subnode, tree=tree)

    return tree


def markov_clustering(*, adj_matrix, species, species_to_index):
    with tempfile.TemporaryDirectory() as tmp:
        triplets_filename = os.path.join(tmp, "triplets.txt")
        tab_filename = os.path.join(tmp, "triplets.tab")
        mci_filename = os.path.join(tmp, "triplets.mci")
        out_filename = os.path.join(tmp, "triplets.out")
        with open(triplets_filename, "w") as temp_triplets:
            temp_triplets.write(
                "\n".join(
                    f"{a} {b} {adj_matrix[species_to_index[a], species_to_index[b]]}"
                    for a, b in itertools.combinations(species, 2)
                    if adj_matrix[species_to_index[a], species_to_index[b]] != 0
                )
            )

        p = subprocess.run(
            [
                "/usr/bin/env",
                "mcxload",
                "-abc",
                triplets_filename,
                "--stream-mirror",
                "-write-tab",
                tab_filename,
                "-o",
                mci_filename,
            ],
            capture_output=True,
        )
        assert p.returncode == 0, p.stderr.decode("utf8")

        p = subprocess.run(
            ["/usr/bin/env", "mcl", mci_filename, "-o", out_filename],
            capture_output=True,
        )
        # noinspection PyUnresolvedReferences
        assert p.returncode == 0, p.stderr.decode("utf8")

        p = subprocess.run(
            [
                "/usr/bin/env",
                "mcxdump",
                "-icl",
                out_filename,
                "-tabr",
                tab_filename,
            ],
            capture_output=True,
        )
        assert p.returncode == 0, p.stderr.decode("utf8")
    lines = p.stdout.decode("utf8").split("\n")
    components = []
    for line in lines:
        if line.startswith("[mclIO]") or line.startswith(".") or len(line.strip()) == 0:
            continue
        parts = line.split()
        if all(part in species for part in parts):
            components.append(list(parts))
    # MCL just omits things like singletons, I guess?
    leftovers = list(set(species).difference(set(reduce(set.union, components, set()))))
    if len(leftovers) > 0:
        components.append(leftovers)
    return components


def spectral_laplacian(*, adj_matrix, species):
    degree = np.sum(adj_matrix, axis=1)
    L = np.diag(degree) - adj_matrix
    evals, evecs = np.linalg.eigh(L)  # right eigenvectors
    evals = evals.real.astype(np.float16).astype(np.float64)
    evecs = evecs.real.astype(np.float16).astype(np.float64)
    special_evals = np.isclose(evals, 0.0)
    num_special_evals = np.sum(special_evals)
    if num_special_evals > 1:
        # When the graph is not connected, first eigenvalues are indicators for components.
        # noinspection PyTupleAssignmentBalance
        L, U = scipy.linalg.lu(
            evecs[:, special_evals].T,
            permute_l=True,
        )
        U = U.real.astype(np.float16)
        component_vec = U[-1, :]
        component_a = np.array(species)[component_vec == 0]
        component_b = np.array(species)[component_vec != 0]
    else:
        idcs = np.argsort(evals)
        second_smallest_idx = idcs[1]
        special_evals = np.isclose(evals, evals[second_smallest_idx])

        if np.sum(special_evals) > 1:
            # noinspection PyTupleAssignmentBalance
            L, U = scipy.linalg.lu(
                evecs[:, special_evals].T,
                permute_l=True,
            )
            U = U.real.astype(np.float16)
            component_vec = U[-1, :]
        else:
            component_vec = evecs[:, second_smallest_idx]

        pos_count = np.sum(component_vec > 0)
        neg_count = np.sum(component_vec < 0)
        # put the zeros with whichever side is smaller, ties to negative side
        if pos_count < neg_count:
            component_a = np.array(species)[component_vec >= 0]
            component_b = np.array(species)[component_vec < 0]
        else:
            component_a = np.array(species)[component_vec > 0]
            component_b = np.array(species)[component_vec <= 0]
    components = [component_a, component_b]
    return components


def markov_p(*, adj_matrix, species):
    degree = np.sum(adj_matrix, axis=1)
    P = np.diag([d**-1 if d != 0 else 0 for d in degree]) @ adj_matrix
    # evals, evecs = np.linalg.eig(P.transpose())  # left eigenvectors
    evals, evecs = np.linalg.eig(P)  # right eigenvectors
    special_evals = np.isclose(evals, 1.0) | np.isclose(evals, 0.0)
    num_special_evals = np.sum(special_evals)
    if num_special_evals > 1:
        # noinspection PyTupleAssignmentBalance
        L, U = scipy.linalg.lu(
            evecs[:, special_evals].T.astype(np.float16).astype(np.float64),
            permute_l=True,
        )
        U = U.real.astype(np.float16).astype(np.float64)
        component_vec = U[-1, :]
        component_a = np.array(species)[component_vec == 0]
        component_b = np.array(species)[component_vec != 0]
    else:
        idcs = np.argsort((evals - 1) ** 2)
        closest_idx_not_one = idcs[1]
        special_evals = np.isclose(evals, evals[closest_idx_not_one])

        if np.sum(special_evals) > 1:
            # noinspection PyTupleAssignmentBalance
            L, U = scipy.linalg.lu(
                evecs[:, special_evals].astype(np.float16).astype(np.float64).T,
                permute_l=True,
            )
            U = U.real.astype(np.float16)
            component_vec = U[-1, :]
        else:
            component_vec = evecs[:, closest_idx_not_one]

        pos_count = np.sum(component_vec > 0)
        neg_count = np.sum(component_vec < 0)
        # put the zeros with whichever side is smaller, ties to negative side
        if pos_count < neg_count:
            component_a = np.array(species)[component_vec >= 0]
            component_b = np.array(species)[component_vec < 0]
        else:
            component_a = np.array(species)[component_vec > 0]
            component_b = np.array(species)[component_vec <= 0]
    components = [component_a, component_b]
    return components


w_tree = gen_tree_weighted(all_triplets, all_weights)

output = w_tree.write(format=9)
if opt.out is None or len(opt.out) == 0:
    print(output, flush=True)
else:
    with open(opt.out, "w") as file:
        file.writelines(output)
