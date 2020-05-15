import argparse
import pickle
import gzip
import sys
import logging
import numpy
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components
import sklearn.cluster
import torch
from models import GraphAutoencoder
import utils
import data

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
    parser.add_argument("-s", "--spec", dest="spec", help="Input file")
    parser.add_argument("-m", "--model", dest="model", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.input, "rb") as ifd:
        outputs = pickle.load(ifd)

    with gzip.open(args.spec, "rb") as ifd:
        spec = pickle.load(ifd)

    with gzip.open(args.model, "rb") as ifd:
        state, margs, _ = torch.load(ifd)

    model = GraphAutoencoder(spec, 
                             margs.depth, 
                             margs.autoencoder_shapes, 
                             margs.embedding_size, 
                             margs.hidden_size, 
                             margs.mask, 
                             margs.field_dropout, 
                             margs.hidden_dropout, 
                             margs.autoencoder)
    model.load_state_dict(state)


    e = {None: 'slave',
         'slave_age': 21.00000049173832,
         'slave_name': 'william',
         'slave_race': 'black',
         'slave_sex': 'm'}
    
    ee = {k : torch.tensor([v]) for k, v in spec.encode(e).items()}
    #print(model.forward(ee, {}))
    #sys.exit()
    #items = {}
    id_to_sources = {}
    id_to_entity = {}
    bottleneck_to_id = {}
    for batch_origs, batch_recons, adjacencies, batch_bottlenecks in outputs:
        batch_recons[None] = batch_origs[None]
        origs = spec.decode_batch(batch_origs)
        recons = spec.decode_batch(batch_recons)
        for i, (orig, recon, bn) in enumerate(zip(origs, recons, batch_bottlenecks)):
            entity_type = orig[None]
            source_rel = "{}_to_source".format(entity_type)
            if source_rel in adjacencies:
                ix = numpy.where(adjacencies[source_rel][i])[0]
                assert(len(ix) == 1)
                src = origs[ix[0]]
            else:
                src = None
            if entity_type != "source":
                bn = tuple(bn.tolist())
                bottleneck_to_id[entity_type] = bottleneck_to_id.get(entity_type, {})
                bottleneck_to_id[entity_type][bn] = bottleneck_to_id[entity_type].get(bn, len(bottleneck_to_id[entity_type]))
                bid = bottleneck_to_id[entity_type][bn]
                id_to_sources[entity_type] = id_to_sources.get(entity_type, {})
                id_to_sources[entity_type][bid] = id_to_sources[entity_type].get(bid, []) + [src]
                id_to_entity[entity_type] = id_to_entity.get(entity_type, {})
                id_to_entity[entity_type][bid] = id_to_entity[entity_type].get(bid, orig)


    id_to_bottleneck = {et : {v : k for k, v in bs.items()} for et, bs in bottleneck_to_id.items()}

    clusters = {}
    for entity_type, i2b in id_to_bottleneck.items():
        #if entity_type not in ["owner"]:
        #    continue
        
        logging.info("Entity type: {}".format(entity_type))
        bottlenecks = numpy.array([i2b[i] for i in range(len(i2b))])
        logging.info("Bottlenecks: {}".format(bottlenecks.shape))
        #clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=round(0.7 * bottlenecks.shape[0]),
        #                                                    linkage="ward").fit(bottlenecks)
        clusterer = sklearn.cluster.KMeans(n_clusters=round(0.7 * bottlenecks.shape[0]),
                                           )
        cids = clusterer.labels_
        clusters[entity_type] = {}
        for i, cid in enumerate(cids):
            clusters[entity_type][cid] = clusters[entity_type].get(cid, [])
            clusters[entity_type][cid].append((id_to_entity[entity_type][i], id_to_sources[entity_type][i]))

    with gzip.open(args.output, "wb") as ofd:
        pickle.dump(clusters, ofd)
