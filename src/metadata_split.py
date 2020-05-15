import argparse
import logging
import data
import gzip
import random
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--entity", dest="entity", help="Entity to split on")
    parser.add_argument("--field", dest="field", help="Metadata field to split on")
    parser.add_argument("--value", dest="value", help="Metadata field value")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    with gzip.open(args.input, "rb") as ifd:
        data = pickle.load(ifd)
    logging.info("Loaded data set: %s", data)
    output_sets = {True : [], False : []}
    for i in range(data.num_components):
        comp, _ = data.component(i)
        vs = set([data._spec.decode(m).get(args.field, None) for m in comp])
        if args.value in vs:
            output_sets[True].append(i)
        else:
            output_sets[False].append(i)
    
    for fname, split in zip(args.outputs, [True, False]):        
        logging.info("Saving %d components to %s", len(output_sets[split]), fname)
        with gzip.open(fname, "wb") as ofd:
            pickle.dump(output_sets[split], ofd)
