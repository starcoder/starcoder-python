import random
import argparse
import pickle
import gzip
import sys
import logging
import numpy
import sklearn.cluster
from sklearn.metrics import f1_score, classification_report, mean_squared_error, mean_absolute_error
import utils
import data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-s", "--spec", dest="spec", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.input, "rb") as ifd:
        outputs = pickle.load(ifd)

    with gzip.open(args.spec, "rb") as ifd:
        spec = pickle.load(ifd)

    cat_fields = []
    num_fields = []
    str_fields = []
    dist_fields = []
    int_fields = []
    for name in spec.field_names:
        if isinstance(spec.field_object(name), data.CategoricalField):
            cat_fields.append(name)
        elif isinstance(spec.field_object(name), data.NumericField):
            num_fields.append(name)
        elif isinstance(spec.field_object(name), data.IntegerField):
            int_fields.append(name)
        elif isinstance(spec.field_object(name), data.DistributionField):
            dist_fields.append(name)            
        elif isinstance(spec.field_object(name), data.SequentialField):
            seq_fields.append(name)            
        
    id_to_sources = {}
    id_to_entity = {}
    bottleneck_to_id = {}
    golds = {}
    guesses = {}
    for batch_origs, batch_recons, adjacencies, batch_bottlenecks in outputs:
        #print(batch_origs)
        #print(batch_recons)
        #sys.exit()
        origs = spec.decode_batch(batch_origs)
        recons = spec.decode_batch(batch_recons)
        for i, (orig, recon) in enumerate(zip(origs, recons)):
            entity_type = orig[spec.entity_type_field]
            for field in [f for f in spec.entity_fields(entity_type) if f in cat_fields + num_fields]:
                golds[field] = golds.get(field, [])
                guesses[field] = guesses.get(field, [])
                golds[field].append(orig[field])
                guesses[field].append(recon[field])

    with open(args.output, "wt") as ofd:
        
        for num_field in num_fields:
            #print(list(zip(golds[num_field], guesses[num_field])))
            
            err = mean_squared_error(golds[num_field], guesses[num_field])
            logging.info("MSE for %s: %.3f", num_field, err)
            ofd.write("{}\tMSE\t{}\n".format(num_field, err))

        for cat_field in cat_fields:
            f1 = f1_score(golds[cat_field], guesses[cat_field], average="macro")
            logging.info("Macro F1 for %s: %.3f", cat_field, f1)
            ofd.write("{}\tF1\t{}\n".format(cat_field, f1))
