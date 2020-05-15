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
from models import GraphAutoencoder, field_models
from data import Dataset
from evaluate import fieldwise
from utils import batchify, stack_batch, split_batch, compute_losses, run_epoch


warnings.filterwarnings("ignore")


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

    logging.basicConfig(level=getattr(logging, args.log_level), 
                        format="%(asctime)s %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")

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
        train_indices = pickle.load(ifd)
    with gzip.open(args.dev, "rb") as ifd:
        dev_indices = pickle.load(ifd)

    train_data = data.subselect(train_indices)
    dev_data = data.subselect(dev_indices)

    model = GraphAutoencoder(data._spec, args.depth, args.autoencoder_shapes, args.embedding_size, args.hidden_size, args.mask, args.field_dropout, args.hidden_dropout, args.autoencoder)

    if args.gpu:
        model.cuda()
        logging.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)
        
    logging.debug("Model: %s", model)
    model.init_weights()
    
    if args.momentum != None:
       logging.info("Using SGD with momentum")
       optim = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    else:
       logging.info("Using Adam")        
       optim = Adam(model.parameters(), lr=args.learning_rate)
    sched = ReduceLROnPlateau(optim, patience=args.patience, verbose=True) if args.patience != None else None

    logging.info("Training StarCoder with %d/%d train/dev entities with %d entities/batch", 
                 len(train_indices), 
                 len(dev_indices), 
                 args.batch_size)

    best_dev_loss = torch.tensor(numpy.nan)
    best_state = None
    since_improvement = 0
    trace = []

    def policy(losses_by_field):
        return sum([x.sum() for x in losses_by_field.values()])

    for e in range(1, args.max_epochs + 1):

        train_loss, train_loss_by_field, dev_loss, dev_loss_by_field = run_epoch(model,
                                                                                 field_models,
                                                                                 optim,
                                                                                 policy,
                                                                                 train_data,
                                                                                 dev_data,
                                                                                 #train_components, 
                                                                                 #dev_components, 
                                                                                 args.batch_size, 
                                                                                 args.gpu)

        trace.append((train_loss, train_loss_by_field, dev_loss, dev_loss_by_field))

        if sched != None:
            sched.step(dev_loss)            

        logging.info("Epoch %d (%d since improvement): train/dev loss = %.4f/%.4f", e, since_improvement, train_loss, dev_loss)

        since_improvement += 1
        if torch.isnan(best_dev_loss) or best_dev_loss > dev_loss:
            logging.info("New best dev loss: %.3f", dev_loss)
            best_dev_loss = dev_loss
            best_state = {k : v.clone().detach().cpu() for k, v in model.state_dict().items()}
            since_improvement = 0
        if args.early_stop != None and since_improvement >= args.early_stop:
            logging.info("Stopping early after no improvement for %d epochs", args.early_stop)
            break

    with gzip.open(args.model_output, "wb") as ofd:
        torch.save((best_state, args, data._spec), ofd)

    with gzip.open(args.trace_output, "wb") as ofd:
        torch.save((trace, args), ofd)
