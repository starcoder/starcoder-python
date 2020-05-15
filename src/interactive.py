import gzip
import pickle
import logging
import argparse
import torch
from data import Dataset, NumericField, DistributionField
from models import GraphAutoencoder
from utils import batchify
import cmd
import ast
from uuid import uuid4
import sys

def to_human(reconstructions, spec):
    lengths = set([v.shape[0] for v in reconstructions.values()])
    assert(len(lengths) == 1)
    length = lengths.pop()
    retval = []
    for i in range(length):
        #entity_type = spec.decode({None : reconstructions[None][i].tolist()})[None]
        item = {} #None : entity_type}

        for k, v in reconstructions.items():
            #if k not in spec.entity_fields(entity_type):
            #    continue
            vals = v[i].tolist()
            if isinstance(vals, int) and vals not in [0, 1]:
                item[k] = vals
            elif isinstance(vals, float):
                item[k] = vals
            elif isinstance(vals, list):
                vals = [x for x in vals if x not in [0, 1]]
                if len(vals) > 0:
                    item[k] = vals
        retval.append(spec.decode(item))
    return retval

def parse_tree(node, prev=None, atype=None):
    u = str(uuid4())
    entity = {"id" : u,
              "entity_type" : "node",
              "name" : str(type(node.op)) if hasattr(node, "op") else "constant",
              }
    if prev != None:
        entity[atype] = prev
    if hasattr(node, "left"):        
        return [entity] + parse_tree(node.left, u, "left_arg") + parse_tree(node.right, u, "right_arg")
    elif hasattr(node, "operand"):
        return [entity] + parse_tree(node.operand, u, "unary_arg")
    elif hasattr(node, "n"):
        entity["value"] = node.n
        return [entity]
    else:
        print("ERROR!")


class Interpreter(cmd.Cmd):
    intro = "Enter mathematical expressions for me to solve (or type 'exit')"
    prompt = ">>> "
    def __init__(self, model, spec):
        self.model = model
        self.spec = spec
        super(Interpreter, self).__init__()
    def default(self, args):
        tree = ast.parse(args, mode="eval")
        entities = parse_tree(tree.body)
        correct = eval(compile(tree, "<AST>", mode="eval"))
        data = Dataset(self.spec, entities)
        for ents, adjs in batchify(data, len(data), subselect=False):
            #encs = [self.spec.encode(e) for e in entities]
            #print(encs)
            #decs = [self.spec.decode(e) for e in encs]
            #print(decs)
            reconstructions, bottlenecks = self.model(ents, adjs)
            reconstructions = {k : (v if isinstance(spec.field_object(k), (DistributionField, NumericField))
                                    else torch.argmax(v, -1, False)) for k, v in reconstructions.items()}
            print(to_human(reconstructions, self.spec))
    def do_exit(self, args):
        return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", help="Input data file")
    parser.add_argument("--split", dest="split", default=None, help="Data split (if none specified, run over everything)")
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")
    parser.add_argument("--batch_size", dest="batch_size", default=128, type=int, help="")
    parser.add_argument("--log_level", dest="log_level", default="INFO", 
                        choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    parser.add_argument("--masked", dest="masked", nargs="*", default=[], help="Fields to mask")
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))

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

    if args.gpu:
        model.cuda()
        logging.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)

    model.train(False)

    interp = Interpreter(model, spec)
    #interp.default("10 + 3")
    interp.cmdloop()
    
    # if args.split == None:
    #     components = [i for i in range(data.num_components)]
    # else:
    #     with gzip.open(args.split, "rb") as ifd:            
    #         components = pickle.load(ifd)

    # outputs = []
    # for i, (entities, adjacencies) in enumerate(batchify(data, components, args.batch_size, subselect=False)):
    #     if args.gpu:
    #         tentities = {k : v.cuda() for k, v in entities.items()}
    #         adjacencies = {k : v.cuda() for k, v in adjacencies.items()}
    #     #mask = tentities["fluency_languages"].sum(1).nonzero().flatten()
    #     #for i in mask:
    #     #    for field in tentities.keys():
    #     #        if field in args.masked:
    #     #            tentities[field][i] = 0
    #     reconstructions, bottlenecks = model(tentities, adjacencies)
    #     reconstructions = {k : (v if isinstance(spec.field_object(k), (DistributionField, NumericField))
    #                             else torch.argmax(v, -1, False)) for k, v in reconstructions.items()}
    #     outputs.append(({k : v.clone().detach().cpu() for k, v in entities.items()}, 
    #                     {k : v.clone().detach().cpu() for k, v in reconstructions.items()}, 
    #                     {k : v.clone().detach().cpu() for k, v in adjacencies.items()}, 
    #                     bottlenecks.cpu()))

    # with gzip.open(args.output, "wb") as ofd:
    #     pickle.dump(outputs, ofd)
