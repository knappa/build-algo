#!/usr/bin/env python3
# implements a weighted BUILD algorithm based on Semple and Steel
import argparse
from functools import reduce
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
from ete3 import Tree

parser = argparse.ArgumentParser()
parser.add_argument("--triplets", required=True, help="triplets file")
parser.add_argument(
    "--out",
    type=str,
    help="output file (Newick format). If unspecified, print to stdout",
)

# try:
opt = parser.parse_args()


#     print(opt)
# except SystemExit:
#     # options for pasting into ipython
#     class Object:
#         pass
#
#     opt = Object()
#     opt.triplets = "test.txt"
#     opt.out = ""


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
    method: Literal["best_random", "spectral", "maxcut"] = "spectral",
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

    species: set = set(reduce(set.union, triplets, set()))
    species_to_index = {s: i for i, s in enumerate(species)}
    adj_matrix = np.zeros((len(species), len(species)), dtype=np.float64)
    degree = np.zeros(len(species), dtype=np.float64)
    for a, b, c in triplets:
        adj_matrix[species_to_index[a], species_to_index[b]] += weights[(a, b, c)]
        degree[species_to_index[a]] += 1
        degree[species_to_index[b]] += 1

    inv_deg = np.diag([d**-1 if d != 0 else 1 for d in degree])

    # noinspection PyPep8Naming
    P = inv_deg @ adj_matrix

    evals, evecs = np.linalg.eig(P)  # right eigenvectors

    idx = np.argmin((np.abs(evals) - 1) ** 2)
    split_vec = evecs[idx]

    component_a = []
    component_b = []
    for idx, spec in enumerate(species):
        if split_vec[idx] <= 0:
            component_a.append(spec)
        else:
            component_b.append(spec)

    for comp in [component_a, component_b]:
        if len(comp) == 1:
            member = list(comp)[0]
            node.add_child(name=member)
        elif len(comp) > 1:
            # filter triplets by component
            comp_triplets = [
                (a, b, c)
                for (a, b, c) in triplets
                if all([x in comp for x in [a, b, c]])
            ]
            subnode = node.add_child()
            if len(comp_triplets) == 0:
                for member in comp:
                    subnode.add_child(name=member)
            else:
                gen_tree_weighted(
                    comp_triplets, weights, node=subnode, tree=tree, method=method
                )

    return tree


w_tree = gen_tree_weighted(all_triplets, all_weights, method="spectral")

output = w_tree.write(format=9)
if opt.out is None or len(opt.out) == 0:
    print(output, flush=True)
else:
    with open(opt.out, "w") as file:
        file.writelines(output)
