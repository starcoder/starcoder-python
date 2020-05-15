import argparse
import pickle
import torch
import gzip
import numpy
from scipy.spatial import distance_matrix
import json
import logging

def read_file(fname):
    with gzip.open(fname, "rb") as ifd:
        return pickle.load(ifd)

def argmax2d(X):
    #print(X[10, 10])
    n, m = X.shape
    x_ = numpy.ravel(X)
    k = numpy.argmin(x_)
    i, j = k // m, k % m
    return i, j


def best_merges(X, n=100):
    nr, nc = X.shape
    #x_ = numpy.ravel(X)
    #x__ = numpy.argsort(x_)
    retval = []
    for i in range(n):
        v = X.argmin()
        r = v // nc
        c = v % nc        
        #print(X[r, c])
        retval.append((r, c))
        #X[r, c] = float("inf")
        X[r, :] = float("inf")
        X[:, c] = float("inf")

        #X[c, r] = float("inf")
        #pass
    #retval = [(i // c, i % c) for i in x__]
    return retval #[(r, c) for r, c in retval if r < c]


def read_entries(ifd, line_count=None):
    entries = {}
    #field_values = {"*entitytype" : set()}
    for line_num, line in enumerate(ifd):
        entry = {}
        if line_count != None and line_num >= line_count:
            break
        j = json.loads(line)
        for field_name, value in j.items():
            entity_type = field_name.split("_")[0]
            entry[entity_type] = entry.get(entity_type, {})                
            entry[entity_type][field_name] = value
            #field_values["*entitytype"].add(entity_type)
            #field_values[field_name] = field_values.get(field_name, set())
            #field_values[field_name].add(value)
        
        key = (entry["source"]["source_file"], entry["source"]["source_sheet"], entry["source"]["source_row"])
        entries[key] = entry
    return entries



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--from", dest="from_file")
    parser.add_argument("--to", dest="to_file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("-d", "--data", dest="data", help="Data file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    with gzip.open(args.data, "rt") as ifd:
        ents = read_entries(ifd)
        
    entity_types = set()
    from_items = {}
#    for outputs, margs, fields in read_file(args.from_file):

    outputs, margs, fields = read_file(args.from_file)
    count = outputs["entitytypes"].shape[0]
    for i in range(count):
        et = fields["*entitytype"][1][1][outputs["entitytypes"][i].item()]
        entity_types.add(et)
        b = outputs["bottlenecks"][i]
        m = outputs["metadata"][i]
        key = (m["source_file"], m["source_sheet"], m["source_row"])
        from_items[et] = from_items.get(et, [])
        from_items[et].append((b, key))

    to_items = {}
    outputs, margs, fields = read_file(args.to_file)
    count = outputs["entitytypes"].shape[0]
    for i in range(count):
        et = fields["*entitytype"][1][1][outputs["entitytypes"][i].item()]
        entity_types.add(et)
        b = outputs["bottlenecks"][i]
        m = outputs["metadata"][i]
        key = (m["source_file"], m["source_sheet"], m["source_row"])
        to_items[et] = to_items.get(et, [])
        to_items[et].append((b, key))

    with gzip.open(args.output, "wt") as ofd:
        for et in entity_types:
            bns = {}
            logging.info("Processing %s entities", et)
            from_bns = numpy.array([x[0].numpy() for x in from_items.get(et, [])])
            to_bns = numpy.array([x[0].numpy() for x in to_items.get(et, [])])
            logging.info("Computing distance between %s entities", et)
            dist = distance_matrix(from_bns, to_bns)
            print(dist.shape)
            logging.info("Finding top merge suggestions for %s entities", et)
            cand = best_merges(dist)
            for r, c in cand:
                ba, keyA = from_items[et][r]
                bb, keyB = to_items[et][c]
                if ents[keyA].get(et, None) != ents[keyB].get(et, None) and et in ents[keyB] and et in ents[keyA]:
                    #if keyA[0] != keyB[0] and ("slavery" in keyA[0] or "slavery" in keyB[0]):
                    #    ofd.write("#####\n")
                    ofd.write("{}\t{}\n".format(et, {k : v for k, v in sorted(ents[keyA].items()) if k in ["slave"]}))
                    ofd.write("{}\t{}\n\n".format(et, {k : v for k, v in sorted(ents[keyB].items()) if k in ["slave"]}))
#                    ofd.write("{}\t{}\n".format(et, {k : v for k, v in sorted(ents[keyA].items()) if k in ["slave", "owner"]}))
#                    ofd.write("{}\t{}\n\n".format(et, {k : v for k, v in sorted(ents[keyB].items()) if k in ["slave", "owner"]}))

            #dist[dist == 0.0] = float("inf")
                #print(bns[fname].shape)
                #empty(shape=len(entities.get(et, [])), 64)
                
        pass
        # for k, v in items.items():
        #     #if k != "slave":
        #     #    continue            
        #     bns = numpy.empty(shape=(len(v), 64))
        #     for i, x in enumerate(v):
        #         bns[i, :] = x[0]
        #     logging.info("Computing distance of %s matrix (%s)", k, bns.shape)
        #     dist = distance_matrix(bns, bns)
        #     dist[dist == 0.0] = float("inf")
        #     #for i in range(len(bns)):
        #     #    dist[i, i] = float("inf")
        #     logging.info("Finding top merge suggestions for %s entities", k)
        #     cand = best_merges(dist)
        #     for a, b in cand:
        #         ba, keyA = items[k][a]
        #         bb, keyB = items[k][b]
        #         if ents[keyA].get(k, None) != ents[keyB].get(k, None) and k in ents[keyB] and k in ents[keyA]:
        #             if keyA[0] != keyB[0] and ("slavery" in keyA[0] or "slavery" in keyB[0]):
        #                 ofd.write("#####\n")
        #             ofd.write("{}\t{}\n".format(k, {k : v for k, v in sorted(ents[keyA].items()) if k in ["slave", "owner"]}))
        #             ofd.write("{}\t{}\n\n".format(k, {k : v for k, v in sorted(ents[keyB].items()) if k in ["slave", "owner"]}))
        #             #ofd.write("{}\t{}\n".format(ents[keyA][k], keyA))
        #             #ofd.write("{}\t{}\n\n".format(ents[keyB][k], keyB))                                    
        #     ofd.write("{}\n\n".format("*" * 30))
        #     #print((fa, sa, ra), (fb, sb, rb))

        # #r, c = argmax2d(dist)
        # #print(r, c, dist[r, c])
        # #print(k, dist.shape)
