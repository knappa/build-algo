#!/usr/bin/env python3
# generate triplets with weights from a true tree
# this is for algorithm testing purposes

import argparse
import itertools
import sys

import ete3
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--newick", required=True, help="newick file")
parser.add_argument("--output", required=True, help="output file")
parser.add_argument(
    "--p", type=float, required=True, help="probability to choose wrong triplet"
)
parser.add_argument("--wt", type=float, default=1.0, help="weight for true triplets")
parser.add_argument("--wf", type=float, default=0.3, help="weight for false triplets")

if hasattr(sys, "ps1"):
    # options for pasting into ipython
    class Object:
        pass

    opt = Object()
    opt.newick = "data/rtree_30_tips_1.nwk"
    opt.output = "test.txt"
    opt.p = 0.01
    opt.wt = 1.0
    opt.wf = 0.3
else:
    opt = parser.parse_args()
    print(opt)

tree = ete3.Tree(opt.newick)
leaves = [leaf for leaf in tree.traverse("postorder") if leaf.is_leaf()]

with open(opt.output, "w") as file:
    for trip in itertools.combinations(leaves, 3):
        subroot = tree.get_common_ancestor(*trip)
        a, b = None, None
        for pair in itertools.combinations(trip, 2):
            if tree.get_common_ancestor(*pair) != subroot:
                a, b = pair
                break
        (c,) = set(trip) - {a, b}
        r = np.random.random()
        if r < opt.p:
            # evenly split likelihood between the two false triplets, when false is chosen
            if r < opt.p / 2:
                file.write(f"{a.name} {c.name} {b.name} {opt.wf}\n")
            else:
                file.write(f"{b.name} {c.name} {a.name} {opt.wf}\n")
        else:
            file.write(f"{a.name} {b.name} {c.name} {opt.wt}\n")
