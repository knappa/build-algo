#!/usr/bin/env python3
# implements a_species weighted BUILD algorithm based on Semple and Steel
import argparse
import itertools
from typing import Dict, List, Literal, Tuple

import networkx as nx
import numpy as np
from ete3 import Tree
from sklearn.cluster import SpectralClustering

parser = argparse.ArgumentParser()
parser.add_argument("--triplets", required=True, help="triplets file")

opt = parser.parse_args()
print(opt)


def get_data(trip_file):
    species = set()
    triplets = list()
    weights = dict()

    with open(trip_file, "r") as file:
        for line in file:
            a, b, c, weight = line.strip().split(" ")
            triplets.append((a, b, c))
            species = species.union([a, b, c])
            weights[(a, b, c)] = weight

    return species, triplets, weights


all_species, all_triplets, all_weights = get_data(opt.triplets)


def gen_tree_weighted(
    triplets: List[Tuple[str, str, str]],
    weights: Dict[Tuple[str, str, str], float],
    *,
    node=None,
    tree=None,
    method: Literal["best_random", "spectral"] = "spectral",
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
            print(f"graph already had an edge between {a} and {b}, summing capacities")
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
        else:  # if method == "spectral":
            # https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering-graphs
            # https://link.springer.com/article/10.1007/s11222-007-9033-z
            adj_mat = nx.to_numpy_array(build_graph, weight="capacity")
            sc = SpectralClustering(
                2, affinity="precomputed", n_init=100, assign_labels="discretize"
            )
            sc.fit(adj_mat)
            nodes_a = list(
                np.array(build_graph.nodes)[np.argwhere(sc.labels_ == 0).reshape(-1)]
            )
            subgraph_a = build_graph.subgraph(nodes_a)
            nodes_b = list(
                np.array(build_graph.nodes)[np.argwhere(sc.labels_ == 1).reshape(-1)]
            )
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
                gen_tree_weighted(comp_triplets, weights, node=subnode, tree=tree)

    return tree


print(gen_tree_weighted(all_triplets, all_weights))
