import pickle
import re
import gzip
import sys
import argparse
import json
import random
import logging
from data import Dataset, Fields


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input-related
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--line_count", dest="line_count", type=int, default=None, help="Only read first L lines")
    parser.add_argument("--max_categorical", dest="max_categorical", type=int, default=5, help="Maximum categorical values")
    parser.add_argument("--max_collapse", dest="max_collapse", default=0, type=int, help="Threshold for treating field-identical entities as the same entity")
    parser.add_argument("--batch_size", dest="batch_size", default=32, help="Batches")
    parser.add_argument("--metadata_prefix", dest="metadata_prefix", default="source", help="Metadata prefix")
    #entity_args = parser.add_mutually_exclusive_group()
    #entity_args.add_argument("--ignore_entity_types", dest="ignore_entity_types", default=[], nargs="+", help="Entity types to ignore")
    parser.add_argument("--keep_entity_types", dest="keep_entity_types", default=[], nargs="+", help="Entity types to keep")
    
    # output-related
    parser.add_argument("--output", dest="output", help="Output file")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.input, "rt") as ifd:
        data = Dataset(ifd, args.line_count, args.max_categorical, args.max_collapse, args.metadata_prefix, args.keep_entity_types)

            
    #remove = [e for e in data._entity_fields.keys() if e not in args.keep_entity_types] if len(args.keep_entity_types) > 0 else args.ignore_entity_types
    #data.remove_entity_types(remove)
    #logging.info("Filtered data set: %s", data)

    data.consistent()
        
    with gzip.open(args.output, "wb") as ofd:
        pickle.dump(data, ofd)
        
    logging.info("Created %s from %s, wrote to %s", data, args.input, args.output)
