import argparse
import gzip
import pickle
import data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.input, "rb") as ifd:
        entity_clusters = pickle.load(ifd)

    with gzip.open(args.output, "wt") as ofd:
        for entity_type, clusters in entity_clusters.items():
            ofd.write("{}\n\n".format(entity_type))
            for cluster in clusters.values():
                for entity, sources in cluster:
                    #for source in sources:
                    ofd.write("{}\n".format(", ".join(["{}={}".format(k, "{:.3}".format(v) if isinstance(v, float) else v) for k, v in entity.items() if k != None])))
                ofd.write("\n")

