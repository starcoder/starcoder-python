import sys
import pickle
import pandas
import json
import starcoder
import argparse
import numpy
import warnings
import gzip
import logging
import torch
from typing import Dict, List, Any, Tuple
from torch.optim import Adam, SGD
from starcoder.schema import Schema
from starcoder.dataset import Dataset
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.registry import field_classes, splitter_classes, batchifier_classes, field_model_classes, scheduler_classes
from starcoder.utils import run_epoch, compute_losses, run_over_components, simple_loss_policy, apply_model, apply_to_components
from starcoder.entity import unstack_entities

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--schema_input", dest="schema_input", help="Input file")
    parser.add_argument("--dataset_input", dest="dataset_input", help="Input file")
    parser.add_argument("--model_input", dest="model_input", help="Input file")
    parser.add_argument("--categorical", nargs="*", default=[], dest="categorical")
    parser.add_argument("--all_categorical", default=False, action="store_true", dest="all_categorical")
    parser.add_argument("--ignore", nargs="*", default=[], dest="ignore")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--early_stop", dest="early_stop", type=int, default=20, help="Early stop")
    parser.add_argument("--patience", dest="patience", type=int, default=5, help="LR scheduler patience (default: no scheduler)")
    parser.add_argument("--max_epochs", dest="max_epochs", default=100, type=int)
    parser.add_argument("--depth", dest="depth", default=2, type=int)
    parser.add_argument("--autoencoder_shapes", dest="autoencoder_shapes", default=[256, 64], type=int)
    parser.add_argument("--schema_output", dest="schema_output", help="Output file")
    parser.add_argument("--dataset_output", dest="dataset_output", help="Output file")
    parser.add_argument("--model_output", dest="model_output", help="Output file")
    parser.add_argument("--reconstruction_output", dest="reconstruction_output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("starcoder")
    logger.setLevel(logging.INFO)

    logger.info("Loading data from '{}'".format(args.input))
    tsv = pandas.read_excel(args.input) if args.input.endswith("xlsx") else pandas.read_csv(args.input)
    tsv = tsv.filter(items=[f for f in tsv.columns if f not in args.ignore])
    known_entity_types: Dict[str, List[str]] = {"row" : []} #"row_number"]} #x[0] : x[1:] for x in args.entity}
    for field_name in tsv.columns:
        entity_type = field_name.split("_")[0]
        known_entity_types[entity_type] = known_entity_types.get(entity_type, []) + [field_name]

    if args.dataset_input:
        with gzip.open(args.dataset_input, "rb") as ifd:
            dataset = pickle.load(ifd)
    else:
        if args.schema_input:
            with open(args.schema_input, "rt") as ifd:
                spec = json.loads(ifd.read())
        else:
            spec = {"meta" : {"entity_type_field" : "entity_type", "id_field" : "id"},
                    "data_fields" : dict([(k, {"type" : "categorical" if (k in args.categorical or args.all_categorical) else "text" if v == "O" else "numeric"}) for k, v in tsv.dtypes.items()] + []),
                    "entity_types" : {k : {"data_fields" : v} for k, v in known_entity_types.items()},
                    "relationship_fields" : {"row_lists_{}".format(k) : {"source_entity_type" : "row", "target_entity_type" : k} for k in known_entity_types.keys() if k != "row"}
            }
        schema = Schema(spec, field_classes)

        entity_lookup: Dict[Tuple[Any, ...], str] = {}
        entities: Dict[str, Dict[str, Any]] = {}
        for row_number in range(len(tsv)):
            row_entities = {}
            for entity_type, fields in known_entity_types.items():
                if entity_type != "row":
                    tuple_entity = tuple([(f, tsv[f][row_number]) for f in fields])
                    if not all([x == "" for _, x in tuple_entity]):
                        eid = entity_lookup.setdefault(tuple_entity, str(len(entity_lookup)))
                        entity = {k : v for k, v in tuple_entity if isinstance(v, str) or not numpy.isnan(v)}
                        entity["id"] = eid
                        row_entities[entity_type] = entity
            row_entities["row"] = dict([("id", "row {}".format(row_number))] + [("row_to_{}".format(e), row_entities[e]["id"]) for e in known_entity_types.keys() if e != "row"])


            for entity_type, entity in row_entities.items():
                entity["entity_type"] = entity_type
                schema.observe_entity(entity)
                entities[entity["id"]] = entity
        schema.minimize()
        schema.verify()
        logging.info("Created schema: %s", schema)
        if args.schema_output != None:
            with open(args.schema_output, "wt") as ofd:
                ofd.write(json.dumps(schema.json))
            logging.info("Wrote schema to '%s'", args.schema_output)


        dataset = Dataset(schema, list(entities.values()), strict=False)
        logging.info("Loaded dataset: %s", dataset)
        if args.dataset_output != None:
            with open(args.dataset_output, "wb") as ofd: # type: ignore
                ofd.write(pickle.dumps(dataset)) # type: ignore
            logging.info("Wrote dataset to '%s'", args.dataset_output)
        
    
    splitter = splitter_classes["sample_components"](["--proportions", ".8", ".1", ".1"])
    train_ids, dev_ids, test_ids = splitter(dataset)
    train_data = dataset.subselect_entities(train_ids)
    dev_data = dataset.subselect_entities(dev_ids)
    model = GraphAutoencoder(schema=dataset.schema,
                             depth=args.depth,
                             autoencoder_shapes=args.autoencoder_shapes, #
                             reverse_relationships=True,
    )
    
    best_dev_loss = torch.tensor(numpy.nan)
    best_state = {}
    logger.info("Model: %s", model)
    logger.info("Model has %d parameters", model.parameter_count)
    model.init_weights()

    optim = Adam(model.parameters(), lr=args.learning_rate)
    sched = scheduler_classes["basic"](args.early_stop, optim, patience=args.patience, verbose=True) # type: ignore
    batchifier = batchifier_classes["sample_components"]([])
    

    logger.info("Training StarCoder with %d/%d train/dev entities and %d/%d train/dev components and batch size %d", 
                len(train_ids), 
                len(dev_ids),
                train_data.num_components,
                dev_data.num_components,
                args.batch_size)
    
    subselect = False
    for e in range(1, args.max_epochs + 1):

        train_loss, train_loss_by_field, dev_loss, dev_loss_by_field = run_epoch(model,
                                                                                 batchifier,
                                                                                 optim,
                                                                                 simple_loss_policy,
                                                                                 train_data,
                                                                                 train_data,
                                                                                 args.batch_size, 
                                                                                 args.gpu,
                                                                                 [],
                                                                                 subselect
        )
        logger.info("Epoch %d: comparable train/dev loss = %.4f/%.4f",
                    e,
                    dev_data.num_entities * (train_loss / train_data.num_entities),
                    dev_loss,
        )

        reduce_rate, es, new_best = sched.step(dev_loss)
        if new_best:
            logger.info("New best dev loss: %.3f", dev_loss)
            best_dev_loss = dev_loss
            best_state = {k : v.clone().detach().cpu() for k, v in model.state_dict().items()}

        if reduce_rate == True:
            model.load_state_dict(best_state)
        if es == True:
            logger.info("Stopping early after no improvement for %d epochs", args.early_stop)
            break
        elif e == args.max_epochs:
            logger.info("Stopping after reaching maximum epochs")
            
    if args.model_output != None:
        logger.info("Saved model to '%s'", args.model_output)

    reconstructions = {}
    for decoded_fields, norm, _ in apply_to_components(model, batchifier_classes["sample_components"]([]), dataset, args.batch_size, args.gpu):
        for entity in [dataset.schema.unpack(e) for e in unstack_entities(norm, dataset.schema)]:
            reconstructions[entity["id"]] = entity

    if args.reconstruction_output:
        with open(args.reconstruction_output, "wt") as ofd:
            for eid, entity in entities.items():
                item = {"original" : entity,
                        "reconstruction" : reconstructions[eid]}
                ofd.write(json.dumps(item) + "\n")
        logger.info("Saved reconstructions to '%s'", args.reconstruction_output)
