import argparse
import gzip
import json
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()
    
    fields = set()
    count = int(len(args.inputs) / 2)
    items = []
    for i in range(count):
        with open(args.inputs[i * 2], "rt") as ifd:
            score = float(ifd.read().strip())
        with gzip.open(args.inputs[i * 2 + 1], "rt") as ifd:
            j = json.load(ifd)
            j["SCORE"] = "{:.3f}".format(score)
        for k, v in j.items():
            fields.add(k)
        items.append(j)
    fields = ["SCORE"] + sorted([f for f in fields if f != "SCORE"])
    with gzip.open(args.output, "wt") as ofd:
        ofd.write("\t".join(fields) + "\n")
        for item in items:
            ofd.write("\t".join(["{}".format(item.get(f, "")) for f in fields]) + "\n")
