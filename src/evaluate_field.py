import random
import argparse
import pickle
import gzip
import sys
import logging
import numpy
import sklearn.cluster
from sklearn.metrics import f1_score
import utils
import data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-s", "--spec", dest="spec", help="Input file")
    parser.add_argument("-f", "--field", dest="field", help="Field to evaluate")
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
    golds = []
    guesses = []
    for batch_origs, batch_recons, adjacencies, batch_bottlenecks in outputs:
        batch_recons[None] = batch_origs[None]
        origs = spec.decode_batch(batch_origs)
        recons = spec.decode_batch(batch_recons)
        for i, (orig, recon, bn) in enumerate(zip(origs, recons, batch_bottlenecks)):
            entity_type = orig[None]
            gold_set = set(orig[args.field].keys())
            guess = recon[args.field]
            if len(gold_set) > 0:
                guess_set = set([x[0] for x in sorted(guess.items(), key=lambda x : x[1], reverse=True)[0:len(gold_set)]])
                cor = list(guess_set & gold_set)
                incor = list(guess_set - gold_set)
                missed = list(gold_set - guess_set)
                # print(gold_set)
                # print(guess_set)
                # print(cor)
                # print(incor)
                # print(missed)
                for c in cor:
                #     print(c, c)
                    golds.append(c)
                    guesses.append(c)
                random.shuffle(incor)
                
                for a, b in zip(incor, missed):
                #     print(a, b)
                    golds.append(b)
                    guesses.append(a)
                # print()

                #cor = 
                #for v in gold:
                #    golds.append(v)
                #    if v in guesses:
                #        
                #print(set(gold.keys()), bguess)
            #sys.exit()
            #related = []
            #for rel_type, adjs in adjacencies.items():
            #    #pass
            #    for j in numpy.where(adjs[i])[0]:
            #        #print(j)
            #        related.append(origs[j])

            #source_rel = "{}_to_source".format(entity_type)            
            #if source_rel in adjacencies:
            #    ix = numpy.where(adjacencies[source_rel][i])[0]
            #    assert(len(ix) == 1)
            #    src = origs[ix[0]]
            #else:
            #    src = None
            #if entity_type != "source":
            #bn = tuple(bn.tolist())
            #bottleneck_to_id[entity_type] = bottleneck_to_id.get(entity_type, {})
            #bottleneck_to_id[entity_type][bn] = bottleneck_to_id[entity_type].get(bn, len(bottleneck_to_id[entity_type]))
            #bid = bottleneck_to_id[entity_type][bn]
            #id_to_sources[entity_type] = id_to_sources.get(entity_type, {})
            #id_to_sources[entity_type][bid] = id_to_sources[entity_type].get(bid, []) + related
            #id_to_entity[entity_type] = id_to_entity.get(entity_type, {})
            #id_to_entity[entity_type][bid] = id_to_entity[entity_type].get(bid, orig)

    print(f1_score(golds, guesses, average="macro"))
    #with gzip.open(args.output, "wb") as ofd:
    #    pickle.dump(clusters, ofd)
