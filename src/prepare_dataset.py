import argparse
import gzip
import pickle
import logging
import json
from data import Dataset, Spec

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_input", dest="schema_input", help="Schema input file")
    parser.add_argument("--data_input", dest="data_input", help="Data input file")
    parser.add_argument("--spec_output", dest="spec_output", help="Spec output file")
    parser.add_argument("--dataset_output", dest="dataset_output", help="Dataset output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.schema_input, "rt") as ifd:
        schema = json.load(ifd)

    field_types = {k : v["type"] for k, v in schema.items()}

    entity_type_fields = [field_name for field_name, field_type in field_types.items() if field_type == "entity_type"]
    assert len(entity_type_fields) == 1, "Exactly one field should be of type 'entity_type'"
    entity_type_field = entity_type_fields[0]

    id_fields = [field_name for field_name, field_type in field_types.items() if field_type == "id"]
    assert len(entity_type_fields) == 1, "Exactly one field should be of type 'id'"
    id_field = id_fields[0]

    # information to gather about possible combinations of entities and relations
    field_values = {k : (v["source_entity_type"], v["target_entity_type"])
                    for k, v in schema.items() if v["type"] == "relation"}
    entity_type_to_fields = {}
    entity_relationship_types = {}
    
    # the actual entities
    entities = {}
    #relations = {}

    with gzip.open(args.data_input, "rt") as ifd:
        for entity in map(json.loads, ifd):
            try:
                entity_type = entity[entity_type_field]
            except:
                print(entity)
                sys.exit()
            entity_id = entity[id_field]
            entity_type_to_fields[entity_type] = entity_type_to_fields.get(entity_type, set())
            entities[entity_id] = entity
            for field_name, field_value in entity.items():
                if field_name not in field_types:
                    continue
                entity_type_to_fields[entity_type].add(field_name)
                if field_types[field_name] not in ["relation"]:
                    field_values[field_name] = field_values.get(field_name, set())
                    field_values[field_name].add(field_value)
                    
    spec = Spec(field_types,
                field_values, 
                entity_type_to_fields)
    #entity_relationship_types)

    logging.info("Created %s", spec)

    dataset = Dataset(spec, entities.values()) #, relations)
    
    with gzip.open(args.spec_output, "wb") as ofd:
        pickle.dump(spec, ofd)
    
    with gzip.open(args.dataset_output, "wb") as ofd:
        pickle.dump(dataset, ofd)
        
    logging.info("Created %s from %s and %s, wrote to %s and %s",
                 dataset,
                 args.data_input,
                 args.schema_input,
                 args.spec_output,
                 args.dataset_output)
