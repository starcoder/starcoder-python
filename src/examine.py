import argparse
import pickle
import gzip
import sys
import logging
import numpy
from scipy.spatial import distance_matrix


def best_merges(X, n=100):
    nr, nc = X.shape
    retval = []
    for i in range(n):
        v = X.argmin()
        r = v // nc
        c = v % nc        
        retval.append((r, c))
        #X[r, :] = float("inf")
        #X[:, c] = float("inf")
        X[r, c] = float("inf")
        X[c, r] = float("inf")
    return retval


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.input, "rb") as ifd:
        outputs = pickle.load(ifd)
    sys.exit()        
    items = {}
    for batch_recons, batch_origs, batch_bottlenecks in outputs:
        for recon, orig, bn in zip(batch_recons, batch_origs, batch_bottlenecks):
            bn = bn.tolist()
            entity_type = orig[None]
            items[entity_type] = items.get(entity_type, [])
            items[entity_type].append((recon, orig, bn))

    for entity_type, entities in items.items():
        if entity_type not in ["slave"]:
            continue
        logging.info("Entity type: {}".format(entity_type))
        bottlenecks = numpy.array([bn for _, _, bn in entities])
        logging.info("Bottlenecks: {}".format(bottlenecks.shape))
        dists = distance_matrix(bottlenecks, bottlenecks)
        logging.info("Distances: {}".format(dists.shape))
        for i in range(dists.shape[0]):
            dists[i, i] = float("inf")
        for first, second in best_merges(dists, 100):
            first = entities[first][1]
            second = entities[second][1]
            first = {k : ("{:.3f}".format(v) if isinstance(v, float) else v) for k, v in first.items() if isinstance(v, (float, str)) and k != None and k.startswith(entity_type)}
            second = {k : ("{:.3f}".format(v) if isinstance(v, float) else v) for k, v in second.items() if isinstance(v, (float, str)) and k != None and k.startswith(entity_type)}
            fields = list(set([k for k in first.keys()] + [k for k in second.keys()]))
            repA = "({})".format(",".join([str(first.get(f, "*")) for f in fields]))
            repB = "({})".format(",".join([str(second.get(f, "*")) for f in fields]))            
            print("  {} <--> {}".format(repA, repB))
        print(fields)
