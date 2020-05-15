import argparse
import gzip
import json
from nltk import Tree

# https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

def process_tree(node, path):
    parent_id = "_".join([str(x) for x in path[:-1]])
    node_id = "_".join([str(x) for x in path])
    if isinstance(node, str):
        return [{"node_id" : node_id, "node_parent_id" : parent_id, "node_text" : node}]
    else:
        current = {"node_id" : node_id, "node_parent_id" : parent_id, "node_sentiment" : int(node.label())}
        return [current] + sum([process_tree(c, path + [i]) for i, c in enumerate(node)], [])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", help="Train file")
    parser.add_argument("--dev", dest="dev", help="Dev file")
    parser.add_argument("--test", dest="test", help="Test file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    trees = []
    for name in ["train", "dev", "test"]:
        with gzip.open(getattr(args, name), "rt") as ifd:
            for i, line in enumerate(ifd):
                tree = Tree.fromstring(line)
                path = [name, i]
                node_id = "_".join([str(x) for x in path])
                top_level = {"node_id" : node_id}
                trees += ([top_level] + process_tree(tree, path + [0]))

    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps(trees, indent=2))
