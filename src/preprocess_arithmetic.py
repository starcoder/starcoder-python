import argparse
import gzip
import json
import random
import ast
from arithmetic import random_tree, generate_constants, generate_tree, populate_tree, flatten_tree, render_tree, grow_tree, reverse_edges, binops, unops

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("-c", "--components", dest="components", type=int, default=1000, help="Number of graph components to generate")
    parser.add_argument("--minimum_constants", dest="minimum_constants", type=int, default=1, help="Minimum number of constants")
    parser.add_argument("--maximum_constants", dest="maximum_constants", type=int, default=4, help="Maximum number of constants")
    parser.add_argument("--unary_probability", dest="unary_probability", type=float, default=0.0, help="Probability of unary ops")
    args = parser.parse_args()

    with gzip.open(args.output, "wt") as ofd:
        for component_id in range(1, args.components + 1):
            for node in random_tree(args.minimum_constants, args.maximum_constants, args.unary_probability, prefix="{}".format(component_id)):
                ofd.write(json.dumps(node) + "\n")

