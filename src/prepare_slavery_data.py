import pickle
import re
import gzip
import sys
import argparse
import json
import random
import logging
from data import Dataset, Spec


field_types = {
    "integer" : ["source_row", "voyage_count", "slave_owner_count", "voyage_manifest_count", "notice_party_size", "slave_age"],
    "categorical" : [None, 'vessel_name', 'slave_sex', 'slave_type', 'source_sheet', 'source_file', 'author_type', 'notice_reward_currency', 'slave_race', 'notice_family_in_baltimore', 'owner_sex', 'gazette_name', 'notice_clothing', 'jail_state', 'jail_name', 'vessel_type', 'notice_headed_for_baltimore'],
    "numeric" : ['notice_date', 'voyage_departure_date', 'notice_reward_amount', 'voyage_manifest_date', 'voyage_duration', 'voyage_arrival_date', 'notice_event_date', 'slave_height', 'vessel_tonnage'],
    "sequential" : ['owner_name', 'voyage_departure_location', 'slave_name', 'owner_location', 'author_name', 'shipper_name', 'slave_alias', 'shipper_location', 'notice_note', 'voyage_arrival_location', 'notice_capture_location', 'captain_name', 'vessel_location', 'consignor_location', 'notice_justification', 'consignor_name'],
}


entity_relationships = {
    "notice" : ["slave", "jail", "author", "gazette"],
    "voyage": ["captain", "vessel", "slave", "shipper", "consignor", "owner"],
    "slave" : ["owner"],
    "source" : ["slave", "jail", "author", "gazette", "captain", "vessel", "shipper", "consignor", "owner", "voyage"]
}

def merge_test(entity_a, entity_b):
    type_a = entity_a[None]
    type_b = entity_b[None]
    name_field_a = "{}_name".format(type_a)
    name_field_b = "{}_name".format(type_b)
    name_a = entity_a.get(name_field_a, "")
    name_b = entity_b.get(name_field_b, "")
    return (type_a == type_b and 
            name_a == name_b and 
            (type_a in ["vessel", "gazette"] or len(re.sub(r"\s+", " ", name_a).strip().split()) > 1))


def entry_filter_test(entry):
    pass

def entity_filter_test(entity):
    pass
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input-related
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--line_count", dest="line_count", type=int, default=None, help="Only read first L lines")
    parser.add_argument("--max_sequence_length", dest="max_sequence_length", 
                        type=int, default=None, help="Max length of sequential fields")
    parser.add_argument("--collapse_identical", dest="collapse_identical", default=False, action="store_true")
    parser.add_argument("--spec_output", dest="spec_output", help="Output file")
    parser.add_argument("--data_output", dest="data_output", help="Output file")
    parser.add_argument("--limit_entity_types", dest="limit_entity_types", nargs="*", help="Only keep entities of the types listed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    field_types = dict(sum([[(f, t) for f in ff] for t, ff in field_types.items()], []))

    field_values = {None : set()}

    num_so_far = 0
    all_entries = []
    all_edges = {}
    entity_type_to_fields = {}
    entity_to_id = {}
    id_to_entity = {}
    entity_relation_types = {}
    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            if i >= args.line_count:
                break
            entity_map = {}
            for field_name, value in json.loads(line).items():
                if value == "" or field_name not in field_types:
                    continue
                if field_types[field_name] == "numeric":
                    value = float(value)
                elif field_types[field_name] == "sequential":
                    value = value[0:args.max_sequence_length if args.max_sequence_length != None else len(value)]
                entity_type, _ = re.match(r"^([^_]+)_(.+)$", field_name).groups()
                #if entity_type not in ["slave", "owner", "source", "voyage"]:
                #    continue
                field_values[None].add(entity_type)
                entity_map[entity_type] = entity_map.get(entity_type, {})                
                entity_map[entity_type][field_name] = value
                field_values[field_name] = field_values.get(field_name, set())
                field_values[field_name].add(value)
                entity_type_to_fields[entity_type] = entity_type_to_fields.get(entity_type, set())
                entity_type_to_fields[entity_type].add(field_name)
            entities = []
            
            # filter entries with missing names or entities
            if entity_map.get("slave", {}).get("slave_name", "") == "" or entity_map.get("owner", {}).get("owner_name", "") == "":
                continue
            for et, vs in [(et, tuple(sorted([(fn, fv) for fn, fv in v.items()]))) for et, v in entity_map.items()]:
                ds = dict(vs)
                eid = entity_to_id.setdefault((et, vs), len(entity_to_id)) if args.collapse_identical else len(id_to_entity)
                id_to_entity[eid] = (et, vs + ((None, et),))
                entities.append((et, eid))
            for se_type, si in entities:
                for linked_type in entity_relationships.get(se_type, []):
                    for te_type, ti in entities:
                        if te_type == linked_type:
                            relation_type = "{}_to_{}".format(se_type, te_type)
                            rev_relation_type = "{}_to_{}".format(te_type, se_type)
                            all_edges[relation_type] = all_edges.get(relation_type, [])
                            all_edges[rev_relation_type] = all_edges.get(rev_relation_type, [])
                            all_edges[relation_type].append((si, ti))
                            all_edges[rev_relation_type].append((ti, si))
                            assert(entity_relation_types.get(relation_type, None) in [None, (se_type, te_type)])
                            entity_relation_types[relation_type] = (se_type, te_type)
                            assert(entity_relation_types.get(rev_relation_type, None) in [None, (te_type, se_type)])
                            entity_relation_types[rev_relation_type] = (te_type, se_type)

    all_entities = [dict(id_to_entity[i][1]) for i in range(len(id_to_entity))]

    spec = Spec({k : v for k, v in field_types.items() if k in field_values}, 
                field_values, 
                entity_type_to_fields,
                entity_relation_types)

    logging.info("Created %s", spec)

    data = Dataset(spec, all_entities, all_edges)

    with gzip.open(args.spec_output, "wb") as ofd:
        pickle.dump(spec, ofd)
    
    with gzip.open(args.data_output, "wb") as ofd:
        pickle.dump(data, ofd)
        
    logging.info("Created %s from %s, wrote to %s and %s", data, args.input, args.spec_output, args.data_output)
