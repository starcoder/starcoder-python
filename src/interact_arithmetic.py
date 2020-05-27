import sys
import gzip
import logging
import argparse
import torch
from data import Dataset, NumericField, DistributionField, IdField, EntityTypeField
from models import GraphAutoencoder
from arithmetic import render_tree, flatten_tree, grow_tree, reverse_edges, populate_tree, to_json
from utils import batchify
import ast
import cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--log_level", dest="log_level", default="INFO", 
                        choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))

    tree = to_json(populate_tree(ast.parse("1+3").body[0], "0"))
    with gzip.open(args.model, "rb") as ifd:
        state, margs, spec = torch.load(ifd)

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
    model.eval()
    model.train(False)

    def one_equation(eq):
        entities = to_json(populate_tree(ast.parse(eq).body[0], "0"))
        correct = {}
        for i in range(len(entities)):
            correct[entities[i]["id"]] = entities[i]["value"]
            if entities[i]["name"] != "const":            
                del entities[i]["value"]
        data = Dataset(spec, entities)
        for i, (ents, adjs) in enumerate(batchify(data, len(entities), subselect=False)):
            reconstructions, bottlenecks = model(ents, adjs)
            reconstructions = {k : (v if isinstance(spec.field_object(k), (DistributionField, NumericField, IdField, EntityTypeField))
                                    else torch.argmax(v, -1, False)) for k, v in reconstructions.items()}        
            for entity in spec.decode_batch(reconstructions):
                #if entity["id"] == "0_1":
                print("Entity {} should be {:.3f}, got {:.3f}".format(entity["id"], correct[entity["id"]], entity["value"]))

    class Interp(cmd.Cmd):
        def default(self, line):
            if line.strip().lower().startswith("q"):
                return True
            else:
                one_equation(line)

    interp = Interp()
    interp.cmdloop("Enter equations consisting of numbers, '+', '-', '*', '/', and parentheses  ('q' to exit)")
