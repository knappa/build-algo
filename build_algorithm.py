#!/usr/bin/env python3

# implements the BUILD algorithm as described in Semple and Steel

import argparse
from typing import List, Tuple

import networkx as nx
from ete3 import Tree

parser = argparse.ArgumentParser()
parser.add_argument("--triplets", required=True, help="triplets file")

opt = parser.parse_args()
print(opt)


def get_data(trip_file):
    species = set()
    triplets = list()

    with open(trip_file, "r") as file:
        for line in file:
            a, b, c, *weight = line.strip().split(" ")
            triplets.append((a, b, c))
            species = species.union([a, b, c])

    return species, triplets


all_species, all_triplets = get_data(opt.triplets)


def gen_tree(triplets: List[Tuple[str, str, str]], *, node=None, tree=None):
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
    for a, b, c in triplets:
        build_graph.add_node(a)
        build_graph.add_node(b)
        build_graph.add_node(c)
        build_graph.add_edge(a, b)
        species = species.union([a, b, c])

    if nx.is_connected(build_graph):
        assert False, f"build graph is connected?\n{build_graph.edges}\n{triplets=}"

    components = list(nx.connected_components(build_graph))

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
                gen_tree(comp_triplets, node=subnode, tree=tree)

    return tree


print(gen_tree(all_triplets))
