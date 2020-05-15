import argparse
import logging
import data
import gzip
import random
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--proportions", dest="proportions", type=float, nargs=3, help="Proportions for train/dev/test splits")
    parser.add_argument("--outputs", dest="outputs", nargs=3, help="Output files for train/dev/test splits")
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if isinstance(args.random_seed, int):
        logging.info("Setting random seed to %d across the board", args.random_seed)
        random.seed(args.random_seed)
    
    with gzip.open(args.input, "rb") as ifd:
        data = pickle.load(ifd)
    logging.info("Loaded data set: %s", data)
    #data.batchify(args.batch_size)
    
    components = [i for i in range(data.num_components)]
    random.shuffle(components)
    train = components[0:int(args.proportions[0] * len(components))]
    dev = components[int(args.proportions[0] * len(components)) : int(sum(args.proportions[0:2]) * len(components))]
    test = components[int(sum(args.proportions[0:2]) * len(components)) : ]


    # tt = data.batchify(train, 32)
    # for x in tt:
    #     e, m, a = data.batch(x)
    #     for k, y in a.items():
    #         for j, z in y.items():
    #             print(k, j, z.shape, z.sum())
                
#        print(a.keys())
        #print(x)
        #data.batch(x, 
    
    for fname, split in zip(args.outputs, [train, dev, test]):
        with gzip.open(fname, "wb") as ofd:
            pickle.dump(split, ofd)
