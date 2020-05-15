from scipy.spatial import distance_matrix

import numpy
import argparse
import logging
import gzip
import pickle
from models import GraphAutoencoder
from flask import Flask, render_template, request
from wtforms import Form, DateField, SelectField, StringField, DecimalField, Field, validators
import torch
from models import GraphAutoencoder
from data import read_entries, unpack_entries, build_auto_spec, compile_data, get_components, batchify


#def human(data_spec, entity_type, field_name, value):
def human(spec, value):    
    field_type = spec[0] #data_spec[entity_type][field_name][0]
    if field_type == "numeric":
        return value.squeeze().item()
    elif field_type == "categorical":
        lu = {v : k for k, v in spec[1].items()}
        return lu[value.item()]
    elif field_type == "sequence":
        lu = {v : k for k, v in spec[1].items()}
        try:
            value = [v.item() for v in value]
        except:
            value = [v.item() for v in value[0]]
        retval = "".join([lu[v] for v in value if v != 1])
        return retval
    else:
        raise Exception()


def unhuman(spec, value):
    field_type = spec[0] #data_spec[entity_type][field_name][0]
    if field_type == "numeric":
        minval, maxval, mean = spec[1:]
        return (mean if value in [None, ""] else float(value))
    elif field_type == "categorical":        
        return spec[1].get(value, 0)
    elif field_type == "sequence":
        seq = [spec[1].get(v, 0) for v in value] if value not in [None, ""] else []
        return (seq + [1] * (spec[-1] - len(seq)))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--annotations", dest="annotations", help="Annotations file")    
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--host", dest="host", default="localhost", help="Host")
    parser.add_argument("--port", dest="port", default=8080, type=int, help="Port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    with gzip.open(args.model, "rb") as ifd:
        state, entity_type_lookup, data_spec, _args = torch.load(ifd, map_location="cpu")
        
    entries, field_values = read_entries(args.input, _args.ignore_json_fields, _args.source_entity_type, _args.line_count)
    entity_to_id, id_to_sources, edges, _ = unpack_entries(entries, _args.ignore_entity_types, _args.source_entity_type, _args.merge_equal)

    components = get_components(len(entity_to_id), edges)
    train, dev, test = [batchify(d, _args.batch_size) for d in compile_data(entity_to_id, data_spec, components, edges, _args.split_proportions, entity_type_lookup, field_values)]
        
        
    model = GraphAutoencoder(data_spec, entity_type_lookup, _args.gcn_depth, _args.autoencoder_shapes)
    model.load_state_dict(state)

    all_entities = {}
    all_bottlenecks = {}
    for entities, adj, mask in dev: #train + dev + test:
        for entity_type, fields in model.reconstruct(entities, adj).items():
            entities = {}
            bottlenecks = {}
            num = len(list([v for k, v in fields.items() if not k.startswith("_")])[0][0])
            for i in range(num):
                entities[i] = {}

                bottlenecks[i] = fields["_bottleneck"][i]
                for field_name, (target, recon) in [x for x in fields.items() if not x[0].startswith("_")]:
                    entities[i][field_name] = (human(data_spec[entity_type][field_name], target[i]), 
                                               human(data_spec[entity_type][field_name], recon[i]))

            for i in range(num): #entity in entities.values():
                all_entities[entity_type] = all_entities.get(entity_type, [])
                all_entities[entity_type].append(entities[i])
                all_bottlenecks[entity_type] = all_bottlenecks.get(entity_type, [])
                all_bottlenecks[entity_type].append(bottlenecks[i].numpy())

    distance_matrices = {}
    desc_sim = {}
    for entity_type, bottlenecks in all_bottlenecks.items():
        bottleneck_matrix = numpy.array(bottlenecks)
        distance_matrices[entity_type] = distance_matrix(bottleneck_matrix, bottleneck_matrix)
        desc_sim[entity_type] = [(a, b) for a, b in [numpy.unravel_index(i, distance_matrices[entity_type].shape) for i in distance_matrices[entity_type].flatten().argsort()] if a < b]
        
    logging.info("Processed %d entities of %d types", sum([len(x) for x in all_entities.values()]), len(all_entities))
    
    app = Flask(__name__, template_folder="templates")

    entity_types = sorted(list(data_spec.keys()))
    modes = [("freeform"),
             ("deduplication"),
             ("clustering"),
             ("browse"),        
             ("field_completion"),
             ("anomaly_detection"),             
    ]
    
    @app.route("/")
    def index():
        return render_template("index.html", entity_types=[k for k in entity_types if k not in ["voyage", "notice"]], modes=modes)

    @app.route("/freeform/<entity_type>", methods=["GET", "POST"])
    def freeform(entity_type):
        default_values = {k : v[3] if v[0] == "numeric" else "<UNK>" if v[0] == "categorical" else "" for k, v in data_spec[entity_type].items()}
        if request.method == "POST":
            entity = {"_type" : torch.tensor([entity_type_lookup[entity_type]])}
            for field_name, spec in data_spec[entity_type].items():
                entity[field_name] = torch.tensor([unhuman(spec, request.form.get(field_name, None))])
            out = model.reconstruct(entity, torch.tensor([1]))
            rentity = list(out.values())[0]
            rentity = {field_name : human(data_spec[entity_type][field_name], v[1]) for field_name, v in rentity.items() if not field_name.startswith("_")}
            #print(rentity)
        return render_template("freeform.html",
                               entity_type=entity_type,
                               entities=all_entities.get(entity_type, []),
                               data_spec=sorted(data_spec.get(entity_type, {}).items()),
                               default_values=default_values
        )

    @app.route("/browse/<entity_type>")
    def browse(entity_type):
        return render_template("browse.html", entity_type=entity_type, entities=all_entities[entity_type])

    @app.route("/clustering/<entity_type>")
    def clustering(entity_type):
        return render_template("clustering.html", entity_type=entity_type, entities=all_entities[entity_type])

    
    @app.route("/field_completion/<entity_type>")
    def field_completion(entity_type):
        return render_template("field_completion.html", entity_type=entity_type, entities=all_entities.get(entity_type, []), data_spec=data_spec.get(entity_type, {}))

    @app.route("/anomaly_detection/<entity_type>")
    def anomaly_detection(entity_type):
        return render_template("anomaly_detection.html", entity_type=entity_type, entities=all_entities.get(entity_type, []), data_spec=data_spec.get(entity_type, {}))

    @app.route("/deduplication/<entity_type>", methods=["GET", "POST"])
    def deduplication(entity_type):
        index = 0
        max_index = len(desc_sim[entity_type]) - 1
        if request.method == "POST":
            index = int(request.form.get("next", 0))
            
            #print(request, index)
        a, b = desc_sim[entity_type][index]
        #print(a, b)
        score = distance_matrices[entity_type][a, b]
        first = {k : v[0] for k, v in all_entities[entity_type][a].items()}
        second = {k : v[0] for k, v in all_entities[entity_type][b].items()}
        fields = sorted(set([x for x in list(first.keys()) + list(second.keys()) if not x.startswith("_")]))
        return render_template("deduplication.html",
                               entity_type=entity_type,
                               first=first,
                               second=second,
                               fields=fields,
                               data_spec=data_spec[entity_type],
                               score=score,
                               index=index,
                               max_index=max_index)
    
    app.run(host=args.host, port=args.port, debug=True)
