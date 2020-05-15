import argparse
import logging
import data
import gzip
import random
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--proportions", dest="proportions", type=float, nargs="+", help="Proportions for splits")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files for splits")
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if isinstance(args.random_seed, int):
        logging.info("Setting random seed to %d across the board", args.random_seed)
        random.seed(args.random_seed)

    assert(len(args.outputs) == len(args.proportions))
    
    with gzip.open(args.input, "rb") as ifd:
        data = pickle.load(ifd)
    logging.info("Loaded data set: %s", data)

    keep_all = []
    to_subselect = []
    for i, d in enumerate(data):
        if d["entity_type"] == "tweet":
            to_subselect.append(i)
        else:
            keep_all.append(i)
            
    #components = [i for i in range(data.num_components)]
    random.shuffle(to_subselect)

    total = sum(args.proportions)
    splits = [(f, int((p / total) * len(to_subselect))) for f, p in zip(args.outputs, args.proportions)]

    for fname, c in splits:
        logging.info("Writing %d entity indices to %s", c, fname)
        with gzip.open(fname, "wb") as ofd:
            pickle.dump(to_subselect[:c] + keep_all, ofd)
        components = to_subselect[c:]
