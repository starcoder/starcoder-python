import argparse
import gzip
import pickle
import logging
import torch
import sacrebleu
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error


def get_scores(name, original, reconstruction, fields):
    of = sorted(list(original.keys()))
    rf = sorted(list(reconstruction.keys()))
    assert(of == rf)
    retval = {}
    for field_name in of:
        assert(original[field_name].shape == reconstruction[field_name].shape)
        if original[field_name].dtype == torch.float32:
            score = 1.0 - mean_absolute_error(original[field_name].detach(), reconstruction[field_name].detach())
        elif original[field_name].shape[1] == 1:
            score = f1_score(original[field_name].detach(), reconstruction[field_name].detach(), average="macro")
        else:
            #score = 0.0
            golds = [" ".join([fields[field_name][1][1][c] for c in l if c != 0]) for l in original[field_name].tolist()]
            guesses = [" ".join([fields[field_name][1][1][c] for c in l if c != 0]) for l in reconstruction[field_name].tolist()]
            #print(golds)
            #print(guesses)
            bleus = []
            for a, b in zip(golds, guesses):
                bleus.append(sacrebleu.sentence_bleu(" ".join(list(a)), " ".join(list(b))))
            score = sum(bleus) / (100.0 * len(bleus))
        #print(field_name, original[field_name].shape)
        key = "{}_{}".format(name, field_name)
        retval[key] = score
    return retval

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    parameter_fields = set()
    score_fields = set()
    items = []
    for fname in args.inputs:
        with gzip.open(fname, "rb") as ifd:
            model_results, model_args, fields = pickle.load(ifd)
            item = {k : v for k, v in vars(model_args).items() if k not in ["data", "dev", "train", "gpu", "log_level", "random_seed"] and not k.endswith("output")}
            for k in item.keys():
                parameter_fields.add(k)
            original = model_results["original"]
            for k, v in [x for x in model_results.items() if not any([y in x[0] for y in ["adjacencies", "swap_values", "edges", "original", "mask"]])]:
                scores = get_scores(k, original, v, fields)
                for k, v in scores.items():
                    score_fields.add(k)                    
                    item[k] = v
                #print(k)
            
            items.append(item)
            #print(model_results.keys())

    field_order = sorted(parameter_fields) + sorted(score_fields)
            
    with gzip.open(args.output, "wt") as ofd:
        ofd.write("\t".join(field_order) + "\n")
        for item in items:
            ofd.write("\t".join(["{}".format(item.get(f, "")) for f in field_order]) + "\n")
