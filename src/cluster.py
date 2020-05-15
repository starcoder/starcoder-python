import argparse
import pickle
import gzip
import sys
import logging
import numpy
import sklearn.cluster
import utils
import data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-s", "--spec", dest="spec", help="Input file")
    parser.add_argument("-r", "--reduction", dest="reduction", type=float, default=0.9, help="")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.input, "rb") as ifd:
        outputs = pickle.load(ifd)

    with gzip.open(args.spec, "rb") as ifd:
        spec = pickle.load(ifd)

    id_to_sources = {}
    id_to_entity = {}
    bottleneck_to_id = {}
    for batch_origs, batch_recons, adjacencies, batch_bottlenecks in outputs:
        batch_recons[None] = batch_origs[None]
        origs = spec.decode_batch(batch_origs)
        recons = spec.decode_batch(batch_recons)
        for i, (orig, recon, bn) in enumerate(zip(origs, recons, batch_bottlenecks)):
            entity_type = orig[None]
            related = []
            for rel_type, adjs in adjacencies.items():
                #pass
                for j in numpy.where(adjs[i])[0]:
                    #print(j)
                    related.append(origs[j])

            #source_rel = "{}_to_source".format(entity_type)            
            #if source_rel in adjacencies:
            #    ix = numpy.where(adjacencies[source_rel][i])[0]
            #    assert(len(ix) == 1)
            #    src = origs[ix[0]]
            #else:
            #    src = None
            #if entity_type != "source":
            bn = tuple(bn.tolist())
            bottleneck_to_id[entity_type] = bottleneck_to_id.get(entity_type, {})
            bottleneck_to_id[entity_type][bn] = bottleneck_to_id[entity_type].get(bn, len(bottleneck_to_id[entity_type]))
            bid = bottleneck_to_id[entity_type][bn]
            id_to_sources[entity_type] = id_to_sources.get(entity_type, {})
            id_to_sources[entity_type][bid] = id_to_sources[entity_type].get(bid, []) + related
            id_to_entity[entity_type] = id_to_entity.get(entity_type, {})
            id_to_entity[entity_type][bid] = id_to_entity[entity_type].get(bid, orig)


    id_to_bottleneck = {et : {v : k for k, v in bs.items()} for et, bs in bottleneck_to_id.items()}

    clusters = {}
    for entity_type, i2b in id_to_bottleneck.items():
        logging.info("Entity type: {}".format(entity_type))
        bottlenecks = numpy.array([i2b[i] for i in range(len(i2b))])
        logging.info("Bottlenecks: {}".format(bottlenecks.shape))
        clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=round(args.reduction * bottlenecks.shape[0]),
                                                            linkage="single").fit(bottlenecks)
        #clusterer = sklearn.cluster.KMeans(n_clusters=round(args.reduction * bottlenecks.shape[0])).fit(bottlenecks)
        cids = clusterer.labels_
        clusters[entity_type] = {}
        for i, cid in enumerate(cids):
            clusters[entity_type][cid] = clusters[entity_type].get(cid, [])
            clusters[entity_type][cid].append((id_to_entity[entity_type][i], id_to_sources[entity_type][i]))

    with gzip.open(args.output, "wb") as ofd:
        pickle.dump(clusters, ofd)
