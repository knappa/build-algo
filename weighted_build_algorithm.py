#!/usr/bin/env python3
# implements a weighted BUILD algorithm based on Semple and Steel
import argparse
import itertools
from typing import Dict, List, Literal, Set, Tuple

import networkx as nx
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


def get_data(trip_file):
    species: Set[str] = set()
    triplets: List[Tuple[str, str, str]] = list()
    weights: Dict[Tuple[str, str, str], float] = dict()

    with open(trip_file, "r") as file:
        for line in file:
            a, b, c, weight = line.strip().split(" ")
            triplets.append((a, b, c))
            species = species.union([a, b, c])
            weights[(a, b, c)] = float(weight)

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

    species = set()
    build_graph = nx.Graph()
    # nx.set_edge_attributes(build_graph, 0, "capacity")
    for a, b, c in triplets:
        build_graph.add_node(a)
        build_graph.add_node(b)
        build_graph.add_node(c)
        capacity = weights[(a, b, c)]
        if build_graph.has_edge(a, b):
            capacity += build_graph.edges[a, b]["capacity"]
            # print(f"graph already had an edge between {a} and {b}, summing capacities")
            # capacity = np.max([capacity, build_graph.edges[a, b]["capacity"]])
        build_graph.add_edge(a, b, capacity=capacity)
        species = species.union([a, b, c])

    components = list(nx.connected_components(build_graph))

    if len(components) == 1:
        if method == "best_random":
            best_cut_size, best_partition = nx.approximation.randomized_partitioning(
                build_graph
            )
            for _ in range(100):
                cut_size, partition = nx.approximation.randomized_partitioning(
                    build_graph
                )
                if cut_size < best_cut_size:
                    best_cut_size = cut_size
                    best_partition = partition
            subgraph_a = build_graph.subgraph(best_partition[0])
            subgraph_b = build_graph.subgraph(best_partition[1])
        elif method == "spectral":
            pos = nx.spectral_layout(build_graph, weight="capacity", dim=1)
            nodes_a = [node for node in build_graph.nodes if pos[node] <= 0]
            subgraph_a = build_graph.subgraph(nodes_a)
            nodes_b = [node for node in build_graph.nodes if pos[node] > 0]
            subgraph_b = build_graph.subgraph(nodes_b)
        else:  # if method == "maxcut":
            # Assemble Triplet MaxCut matrices
            good = np.zeros((len(species), len(species)), dtype=np.float64)
            bad = np.zeros((len(species), len(species)), dtype=np.float64)
            species_to_idx = {sp: idx for idx, sp in enumerate(species)}
            for a, b, c in triplets:
                weight = weights[(a, b, c)]
                a_idx, b_idx, c_idx = map(lambda x: species_to_idx[x], (a, b, c))
                # In Triplet MaxCut algo, the cherry is the "bad" one. For some reason.
                bad[a_idx, b_idx] += weight
                bad[b_idx, a_idx] += weight
                # other pairs are "good"
                good[a_idx, c_idx] += weight
                good[c_idx, a_idx] += weight
                good[b_idx, c_idx] += weight
                good[c_idx, b_idx] += weight

            embedding = np.random.normal(size=(len(species), 3))
            while np.any(np.all(embedding == 0.0, axis=1)):
                embedding = np.random.normal(size=(len(species), 3))
            embedding /= np.linalg.norm(embedding, axis=1)
            embedding = np.nan_to_num(embedding)

            for _ in range(10):
                cm = np.zeros((len(species), 3))
                weights = np.zeros(len(species))
                for i, j in itertools.combinations(range(len(species)), 2):
                    cm[i, j] += embedding[j] * (good[i, j] - 3 * bad[i, j])
                    weights[i] += good[i, j] - 3 * bad[i, j]

                embedding[:, :] = cm / weights
                embedding = np.nan_to_num(embedding)
                embedding[
                    np.linalg.norm(embedding, axis=1) == 0, :
                ] += 1  # fix any zero norm (low prob event)
                embedding /= np.linalg.norm(embedding, axis=1)
                embedding = np.nan_to_num(embedding)

            normal_vec = np.random.uniform(3)
            while np.all(normal_vec == 0.0):
                normal_vec = np.random.uniform(3)
            normal_vec /= np.linalg.norm(normal_vec)

            nodes_a = [
                sp
                for sp in species
                if normal_vec @ embedding[species_to_idx[sp], :] >= 0
            ]
            subgraph_a = build_graph.subgraph(nodes_a)
            nodes_b = [
                sp
                for sp in species
                if normal_vec @ embedding[species_to_idx[sp], :] < 0
            ]
            subgraph_b = build_graph.subgraph(nodes_b)

        components = list(
            itertools.chain(
                nx.connected_components(subgraph_a),
                nx.connected_components(subgraph_b),
            )
        )

    for comp in components:
        if len(comp) == 1:
            member = list(comp)[0]
            node.add_child(name=member)
        else:
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


tree = gen_tree_weighted(all_triplets, all_weights, method="spectral")

output = tree.write(format=9)
if opt.out is None or len(opt.out) == 0:
    print(output, flush=True)
else:
    with open(opt.out, "w") as file:
        file.writelines(output)
