import argparse
import json
import gzip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", dest="schema", help="Schema file")
    parser.add_argument("--data", dest="data", help="Data file")
    args = parser.parse_args()

    
