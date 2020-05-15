import argparse
import csv
import gzip
from plotnine import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    simple, entitywise, typewise, edgewise = args.inputs

    score_fields = set()
    other_fields = set()
    with gzip.open(simple, "rt") as ifd:
        reader = csv.DictReader(ifd, delimiter="\t")
        for row in reader:
            for k, v in row.items():
                if k.endswith("_score"):
                    score_fields.add(k)
                else:
                    other_fields.add(k)
    print(score_fields, other_fields)
