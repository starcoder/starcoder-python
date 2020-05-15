import sys
import argparse
import os
import csv
import gzip
import pickle
import torch
import pandas
from plotnine import *
from matplotlib import pyplot
import tempfile
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

score_types = ["simple"]
group_by = ["depth"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")    
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    field_values = {}
    items = []
    score_fields = set()
    other_fields = set()
    with gzip.open(args.input, "rt") as ifd:
        reader = csv.DictReader(ifd, delimiter="\t")
        for row in reader:
            row = {k : float(v) if k.endswith("_score") else v for k, v in row.items()}
            items.append(row)
            for k, v in row.items():
                field_values[k] = field_values.get(k, set())
                field_values[k].add(v)
                if k.endswith("_score"):
                    score_fields.add(k)
                else:
                    other_fields.add(k)
    df = pandas.DataFrame(items)
    for k, v in field_values.items():
        if len(v) == 1 and not k.endswith("score"):
            #print(k)
            #print(k, df.columns)
            df = df.drop(columns=k)
        elif k.endswith("score"):
            df = df.assign(**{k[7:len(k) - 6] : df[k]})
            df = df.drop(columns=k)
    #df = df[(df.split=="dev")]
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(str(df))
    #print(df)
    sys.exit()
    figures = []
    try:
        _, tname = tempfile.mkstemp(suffix=".png")
        print(tname)
        for score_field in score_fields:
            plot = ggplot(aes("epoch", score_field), data=df) + geom_line(aes(color="factor(depth)"))
            plot.save(tname)
            figures.append(Image.open(tname))
            figures[-1].load()
    except Exception as err:
        raise err
    finally:
        os.remove(tname)
    mode = figures[0].mode
    max_x = max([ex.size[0] for ex in figures])
    max_y = max([ex.size[1] for ex in figures])
    im = Image.new(mode, (max_x, max_y * len(figures)))
    for i, v in enumerate(figures):
        v = v.resize((max_x, max_y))
        im.paste(v, box=(0, i * max_y))
        pass
    im.save(args.output)
