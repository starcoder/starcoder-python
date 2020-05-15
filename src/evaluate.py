import gzip
import pickle
import logging
import argparse
import torch
import sacrebleu
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error
from data import Dataset
from models import GraphAutoencoder

def fieldwise(originals, reconstructions, fields):
    scores = {}
    seq_scores, cat_scores, num_scores = [], [], []
    for key, orig_vals in originals.items():
        recon_vals = torch.cat(reconstructions[key], 0)
        orig_vals = torch.cat(orig_vals, 0)
        entity_type, field_name = key
        if len(recon_vals.shape) == 1:
            # numeric
            golds = orig_vals.squeeze(1).tolist()
            guesses = recon_vals.tolist()
            try:
                score = 1.0 - mean_absolute_error(golds, guesses)
            except:
                score = 0.0
            num_scores.append(score)
        elif len(recon_vals.shape) == 2:
            # categorical
            golds = orig_vals.squeeze(1).tolist()
            guesses = recon_vals.argmax(1).tolist()
            score = f1_score(golds, guesses, average="macro")
            cat_scores.append(score)
        elif len(recon_vals.shape) == 3:
            # sequence            
            golds = ["".join([fields[field_name][1][1][c] for c in l if c != 0]) for l in orig_vals.tolist()]
            guesses = ["".join([fields[field_name][1][1][c] for c in l if c != 0]) for l in recon_vals.argmax(2).tolist()]
            bleus = []
            for a, b in zip(golds, guesses):
                bleus.append(sacrebleu.sentence_bleu(" ".join(list(a)), " ".join(list(b))))
            score = sum(bleus) / (100.0 * len(bleus))
            seq_scores.append(score)
            
        scores[field_name] = score
#     logging.info("""Averages:
#    Numeric 1 - MAE:    %.3f
#    Categorical F1:     %.3f
#    Sequence BLEU:      %.3f
# """, 
#                  0.0 if len(num_scores) == 0 else sum(num_scores) / len(num_scores),
#                  0.0 if len(cat_scores) == 0 else sum(cat_scores) / len(cat_scores),
#                  0.0 if len(seq_scores) == 0 else sum(seq_scores) / len(seq_scores),
#     )
    return scores


def simple_evaluate(model, data, batches, gpu):
    old_state = model.training
    model.train(False)
    loss = torch.tensor(0.0)
    data_len = 0
    outs = {}
    originals, reconstructions = {}, {}
    for i, b in enumerate(batches):
        entities, masks, adjacencies = data.batch(b)
        data_len += list(masks.values())[0].shape[0]
        entities = {k : torch.tensor(v) for k, v in entities.items()}
        masks = {k : torch.tensor(v).type(torch.ByteTensor) for k, v in masks.items()}
        adjacencies = {source_type : {dest_type : torch.tensor(v) for dest_type, v in dests.items()} for source_type, dests in adjacencies.items()}
        
        if gpu:
            entities = {k : v.cuda() for k, v in entities.items()}
            masks = {k : v.cuda() for k, v in masks.items()}
            adjacencies = {source_type : {target_type : m.cuda() for target_type, m in targets.items()} for source_type, targets in adjacencies.items()}
            
        l, rs, os, lba = model(entities, masks, adjacencies)
        loss += l.detach()
        for entity_type, fields in rs.items():
            for field_name, values in fields.items():
                key = (entity_type, field_name)
                reconstructions[key] = reconstructions.get(key, [])
                reconstructions[key].append(values.detach())
        for entity_type, fields in os.items():
            for field_name, values in fields.items():
                key = (entity_type, field_name)
                originals[key] = originals.get(key, [])
                originals[key].append(values.detach())            
        
    model.train(old_state)
    scores = fieldwise(originals, reconstructions, model._data._fields)
    return scores


def entitywise_evaluate(model, data, ids, gpu):
    return {}

def typewise_evaluate(model, data, ids, gpu):
    return {}

def edgewise_evaluate(model, data, ids, gpu):
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    with gzip.open(args.input, "rb") as ifd:
        results, margs = pickle.load(ifd)
    margs = vars(margs)

    values = {k : {} for k in ["original", "simple", "mask_values", "mask_fields", "mask_edges", "swap_values", "inject_edges"]}
    for i in range(len(results["simple"])):
        
        pass

