import logging
from jsonpath_ng.ext import parse
import json
import os.path

logger = logging.getLogger(__name__)


def expand(schema, config=[]):
    for spec_type in ["properties", "entity_types", "relationships"]:
        schema[spec_type] = [{k : v for k, v in list(vals.items()) + [("name", name)]} for name, vals in schema.get(spec_type, {}).items()]
    for pattern_string, values in config:
        pattern = parse(pattern_string)
        for match in pattern.find(schema):
            for submatch in match.value if isinstance(match.value, list) else [match.value]:
                submatch["meta"] = submatch.get("meta", {})
                for k, v in values.items():
                    submatch["meta"][k] = v
    for spec_type in ["properties", "entity_types", "relationships"]:
        schema[spec_type] = {vals["name"] : {k : v for k, v in vals.items() if k != "name"} for vals in schema.get(spec_type, [])}
    return schema


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", dest="schema", help="Schema file")
    parser.add_argument(dest="config_files", nargs="+", help="Configuration files")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with open(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())

    config_rules = []
    for conf_file in args.config_files:
        if os.path.exists(conf_file):
            with open(conf_file, "rt") as ifd:
                config_rules += json.loads(ifd.read())
        else:
            config_rules += eval(conf_file)

    expanded_schema = expand(schema, config_rules)
    expanded_schema["entity_type_property"] = schema["meta"]["entity_type_property"]
    expanded_schema["id_property"] = schema["meta"]["id_property"]

    with open(args.output, "wt") as ofd:
        ofd.write(json.dumps(expanded_schema, indent=2))
