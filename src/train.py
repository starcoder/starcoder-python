import pickle
import re
import gzip
import sys
import argparse
import json
import random
import logging
import warnings
import numpy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from models import GraphAutoencoder, init_weights
from data import Dataset
from evaluate import fieldwise

warnings.filterwarnings("ignore")


def run_epoch(model, data, batches, optim, train=False, gpu=False):
    old_state = model.training
    model.train(train)
    epoch_loss = torch.tensor(0.0)
    data_len = 0
    random.shuffle(batches)
    outs = {}
    #originals, reconstructions = {}, {}
    loss_by_architecture = {}
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
            
        if train:
            optim.zero_grad()
        loss, reconstruction, os, lba, _ = model(entities, masks, adjacencies)
        for k, v in lba.items():
            loss_by_architecture[k] = loss_by_architecture.get(k, [])
            loss_by_architecture[k] += v
        if train:
            loss.backward()
            optim.step()
        epoch_loss += loss.detach()
    loss_by_architecture = {k : sum(v) / len(v) for k, v in loss_by_architecture.items()}
    #print(loss_by_architecture)
        #for entity_type, fields in rs.items():
        #    for field_name, values in fields.items():
        #        key = (entity_type, field_name)
        #        reconstructions[key] = reconstructions.get(key, [])
        #        reconstructions[key].append(values.detach())
        #for entity_type, fields in os.items():
        #    for field_name, values in fields.items():
        #        key = (entity_type, field_name)
        #        originals[key] = originals.get(key, [])
        #        originals[key].append(values.detach())            
        
    model.train(old_state)
    #loss_by_architecture = {k : sum(v) / len(v) for k, v in loss_by_architecture.items()}
    return (epoch_loss / data_len, loss_by_architecture)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input-related
    parser.add_argument("--data", dest="data", help="Input data file")
    parser.add_argument("--train", dest="train", help="Train split components file")
    parser.add_argument("--dev", dest="dev", help="Dev split components file")
    
    # model-related
    parser.add_argument("--depth", dest="depth", type=int, default=0, help="Graph-structure depth to consider")
    parser.add_argument("--embedding_size", type=int, dest="embedding_size", default=32, help="Size of embeddings")
    parser.add_argument("--hidden_size", type=int, dest="hidden_size", default=32, help="Size of embeddings")
    parser.add_argument("--hidden_dropout", type=float, dest="hidden_dropout", default=0.0, help="Size of embeddings")
    parser.add_argument("--field_dropout", type=float, dest="field_dropout", default=0.0, help="Size of embeddings")
    parser.add_argument("--autoencoder_shapes", type=int, default=[], dest="autoencoder_shapes", nargs="*", help="Autoencoder layer sizes")
    parser.add_argument("--mask", dest="mask", default=[], nargs="*", help="Fields to mask")
    parser.add_argument("--autoencoder", dest="autoencoder", default=False, action="store_true", help="Use autoencoder loss directly")
    
    # training-related
    parser.add_argument("--max_epochs", dest="max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--patience", dest="patience", type=int, default=None, help="LR scheduler patience (default: no scheduler)")
    parser.add_argument("--early_stop", dest="early_stop", type=int, default=None, help="Early stop")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", dest="momentum", type=float, default=None, help="Momentum for SGD (default: Adam)")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")
    
    # output-related
    parser.add_argument("--model_output", dest="model_output", help="Model output file")
    parser.add_argument("--trace_output", dest="trace_output", help="Trace output file")

    # miscellaneous
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if isinstance(args.random_seed, int):
        logging.info("Setting random seed to %d across the board", args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        numpy.random.seed(args.random_seed)

    with gzip.open(args.data, "rb") as ifd:
        data = pickle.load(ifd)
    logging.info("Loaded data set: %s", data)

    with gzip.open(args.train, "rb") as ifd:
        train_components = pickle.load(ifd)
    with gzip.open(args.dev, "rb") as ifd:
        dev_components = pickle.load(ifd)
    
    model = GraphAutoencoder(data, args.depth, args.autoencoder_shapes, args.embedding_size, args.hidden_size, args.mask, args.field_dropout, args.hidden_dropout, args.autoencoder)

    if args.gpu:
        model.cuda()
        logging.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)
        
    logging.debug("Model: %s", model)
    init_weights(model)

    if args.momentum != None:
        logging.info("Using SGD with momentum")
        optim = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    else:
        logging.info("Using Adam")        
        optim = Adam(model.parameters(), lr=args.learning_rate)
    sched = ReduceLROnPlateau(optim, patience=args.patience, verbose=True) if args.patience != None else None

    logging.info("Training AutoencodingGraphEnsemble with %d/%d train/dev components batched with max of %d entities", len(train_components), len(dev_components), args.batch_size)
    best_dev_loss = torch.tensor(numpy.nan)
    best_state = None
    since_improvement = 0

    all_scores = {}
    trace = []            
    for e in range(1, args.max_epochs + 1):
        train_batches = data.batchify(train_components, args.batch_size)
        dev_batches = data.batchify(dev_components, args.batch_size)
        train_loss, train_loss_by_architecture = run_epoch(model, data, train_batches, optim, True, args.gpu)
        dev_loss, dev_loss_by_architecture = run_epoch(model, data, dev_batches, None, False, args.gpu)
        trace.append((dev_loss_by_architecture, train_loss_by_architecture))
        if sched != None:
            sched.step(dev_loss)
        logging.info("Epoch %d: train/dev loss = %.4f/%.4f", e, train_loss, dev_loss)
        since_improvement += 1
        if torch.isnan(best_dev_loss) or best_dev_loss > dev_loss:
            logging.info("New best dev loss: %.3f", dev_loss)
            best_dev_loss = dev_loss
            best_state = {k : v.clone().detach() for k, v in model.state_dict().items()}
            since_improvement = 0
        if args.early_stop != None and since_improvement >= args.early_stop:
            logging.info("Stopping early after no improvement for %d epochs", args.early_stop)
            break

    with gzip.open(args.model_output, "wb") as ofd:
        torch.save((best_state, data, args), ofd)

    with gzip.open(args.trace_output, "wb") as ofd:
        torch.save((trace, args), ofd)
