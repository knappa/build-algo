#!/usr/bin/env python3
# the BUILD algorithm as described in Semple and Steel


def main_cli():
    import argparse

    from build_algo import _gen_tree, get_triplets_from_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", required=True, help="triplets file")

    opt = parser.parse_args()
    # print(opt)

    all_species, all_triplets = get_triplets_from_file(opt.triplets)

    print(_gen_tree(all_triplets))
