import argparse
import logging
import os.path
import json
import gzip
import re
import datetime
import calendar
import csv



if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(dest="inputs", nargs="+")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for fname in args.inputs:
        with gzip.open(fname, "rt") as ifd:
            for row in csv.DictReader(ifd):
                pass
