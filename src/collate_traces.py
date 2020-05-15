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
    
    type_fields = set()
    parameter_fields = set(["split", "epoch"])
    items = []
    for fname in args.inputs:
        with gzip.open(fname, "rb") as ifd:
            traces, model_args = torch.load(ifd)
            exp = {k : v for k, v in vars(model_args).items() if k not in ["data", "dev", "train", "gpu", "log_level", "random_seed"] and not k.endswith("output")}
            for k in exp.keys():
                parameter_fields.add(k)            
            for epoch, (dev_loss, train_loss) in enumerate(traces):
                for split_name, split in zip(["dev", "train"], [dev_loss, train_loss]):
                    scores = {"split" : split_name, "epoch" : epoch}
                    for field_type, score in split.items():
                        #if len(score) == 0:
                        #    continue
                        field_type = "{}_loss".format(field_type)
                        scores[field_type] = score #sum(score) / len(score)
                        type_fields.add(field_type)
                    items.append({k : v for k, v in list(exp.items()) + list(scores.items())})

    field_order = sorted(parameter_fields) + sorted(type_fields)
            
    with gzip.open(args.output, "wt") as ofd:
        ofd.write("\t".join(field_order) + "\n")
        for item in items:
            ofd.write("\t".join(["{}".format(item.get(f, "")) for f in field_order]) + "\n")
