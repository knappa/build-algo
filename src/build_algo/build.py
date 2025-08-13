#!/usr/bin/env python3
# the BUILD algorithm as described in Semple and Steel


def main_cli():
    import argparse

    from build_algo import gen_tree, get_triplets_from_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", required=True, help="triplets file")

    parser.add_argument("--viz", action="store_true", help="print visualization")

    opt = parser.parse_args()
    # print(opt)

    all_species, all_triplets = get_triplets_from_file(opt.triplets)

    tree = gen_tree(all_triplets)

    if opt.viz:
        print(tree)
    else:
        print(tree.write(format=9))
