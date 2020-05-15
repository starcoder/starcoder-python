import argparse
import gzip
import pickle
from models import GraphAutoencoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.input, "rb") as ifd:
        state, field_spec, gcn_depth, autoencoder_shapes = pickle.load(ifd)

    model = GraphAutoencoder(field_spec, gcn_depth, autoencoder_shapes)
    model.load_state_dict(state)

    print(field_spec)
    #print(model)
