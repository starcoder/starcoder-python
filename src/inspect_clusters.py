import argparse
import gzip
import pickle
import data
from datetime import date

def format_entity(ent):
    return ", ".join(["{}={}".format(k, date.fromordinal(round(v)).isoformat() if k.endswith("date") else "{:.3}".format(v) if isinstance(v, float) else v) for k, v in ent.items() if k != None])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-e", "--entity_type", help="Only process the given entity type")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.input, "rb") as ifd:
        entity_clusters = pickle.load(ifd)

    with open(args.output, "wt") as ofd:
        for entity_type, clusters in entity_clusters.items():
            if entity_type == "source" or (args.entity_type not in [None, entity_type]):
                continue
            ofd.write("{}\n\n".format(entity_type))
            for cluster in clusters.values():
                for entity, sources in cluster:
                    ofd.write("{}\n".format(format_entity(entity)))
                    #for source in sources:
                    #    ofd.write("\t{}\n".format(format_entity(source)))
                ofd.write("\n")

