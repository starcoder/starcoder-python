#from starcoder.entity import EntityType, UnpackedEntity, PackedEntity
#from starcoder.property import Property, EntityTypeProperty, IdProperty, PackedValueType, UnpackedValueType
#from typing import List, Dict, Any, Set, cast, MutableSequence, Type
import logging
from jsonpath_ng.ext import parse
import json

logger = logging.getLogger(__name__)

# class Schema(object):
#     """
# A Schema object represents and tracks entity-relationship information.

# An instance should be treated as, essentially, three dictionaries
# which map from strings (names) to objects:

#   entity_types, properties, relationships

# and two properties:

#   id_property, entity_type_property

# Schema objects also store the original JSON object they were
# constructed from in the "json" property.
#     """
#     def __init__(self, spec: Dict[str, Any], property_classes: Dict[str, Type[Property]]) -> None: # this "Any" is OK
#         self.json = spec.copy()
#         self.id_property: IdProperty = cast(IdProperty, property_classes["id"](spec["meta"]["id_property"]))
#         self.entity_type_property: EntityTypeProperty = cast(EntityTypeProperty, property_classes["entity_type"](spec["meta"]["entity_type_property"]))
#         self.properties: Dict[str, Property[Any, Any, Any]] = {} # these "Any"s are OK
#         self.relationships = {} #: Dict[str, Relationship] = {}
#         self.entity_types = {}
#         self.seen_entity_types: Set[UnpackedValueType] = set()
#         for property_name, property_spec in spec["properties"].items():
#             if property_spec.get("ignore", False) == False:
#                 property_type = property_spec["type"]
#                 self.properties[property_name] = property_classes[property_type](property_name, **property_spec)
#                 #cast(Property[Any, Any, Any], property_classes[property_type](property_name, **property_spec)) # these "Any"s are OK
#         for property_name, property_spec in spec["relationships"].items():
#             if property_spec.get("ignore", False) == False:
#                 self.relationships[property_name] = property_classes["relationship"](property_name, type="relationship", **property_spec)
#         for entity_type, entity_spec in spec["entity_types"].items():
#             if entity_spec.get("ignore", False) == False:
#                 rel_properties = set([k for k, v in self.relationships.items() if v.source_entity_type == entity_type])
#                 rev_rel_properties = set([k for k, v in self.relationships.items() if v.target_entity_type == entity_type])
#                 properties = set([k for k in entity_spec.get("properties", []) if k in self.properties])
#                 self.entity_types[entity_type] = EntityType(entity_type,
#                                                             properties,
#                                                             rel_properties,
#                                                             rev_rel_properties)

#     def pack(self, entity: UnpackedEntity) -> PackedEntity:
#         entity = {k : entity.get(k, None) for k in self.all_property_names} # type: ignore
#         retval = {k : self.properties[k].pack(v) if k in self.properties else v for k, v in entity.items()}
#         return retval

#     def unpack(self, entity: PackedEntity) -> UnpackedEntity:
#         retval = {k : self.properties[k].unpack(v) if k in self.properties else v for k, v in entity.items()}
#         #reveal_type(retval)
#                   #if k in self.properties for k, v in entity.items()} # else v for k, v in entity.items()}
#         return {k : v for k, v in retval.items() if v not in [[], None]} # type: ignore

#     @property
#     def all_property_names(self) -> List[str]:
#         return list(self.properties.keys()) + list(self.relationships.keys()) + [self.id_property.name, self.entity_type_property.name]

#     @property    
#     def all_property_objects(self) -> List[Property]:
#        return cast(List[Property], list(self.properties.values())) + cast(List[Property], list(self.relationships.values())) + cast(List[Property], [self.id_property, self.entity_type_property])
    
#     def observe_entity(self, entity: UnpackedEntity) -> None:
#         if entity[self.entity_type_property.name] in self.entity_types:
#             self.seen_entity_types.add(entity[self.entity_type_property.name])
#             for k, v in entity.items():
#                 if k in self.properties:
#                     self.properties[k].observe_value(v)

#     def verify(self) -> bool:
#         return True
#         #raise Exception("Property '{}' had no observed values".format(name))

#     def minimize(self) -> None:
#         #print(self.seen_entity_types)
#         # remove properties that haven't been seen
#         for property_name in list(self.properties.keys()):
#             if self.properties[property_name].empty == True:
#                 #print(property_name)
#                 del self.properties[property_name]
#         for entity_type_name in list(self.entity_types.keys()):
#             # remove undefined properties from entity-types
#             self.entity_types[entity_type_name].properties = [f for f in self.entity_types[entity_type_name].properties if f in self.properties]
#             # remove entity-types that haven't been seen
#             if entity_type_name not in self.seen_entity_types:
#                 del self.entity_types[entity_type_name]

#         # remove properties that don't occur in an entity-type
#         for property_name in list(self.properties.keys()):
#             if all([property_name not in e.properties for e in self.entity_types.values()]):
#                 #print(property_name)
#                 del self.properties[property_name]
#         # remove relationships for which an entity-type doesn't exist
#         for property_name in list(self.relationships.keys()):
#             rel_property = self.relationships[property_name]
#             if rel_property.source_entity_type not in self.entity_types or rel_property.target_entity_type not in self.entity_types:
#                 del self.relationships[property_name]
#         for entity_type_name in list(self.entity_types.keys()):
#             rel_properties = set([k for k, v in self.relationships.items() if v.source_entity_type == entity_type_name])
#             rev_rel_properties = set([k for k, v in self.relationships.items() if v.target_entity_type == entity_type_name])
#             self.entity_types[entity_type_name].relationships = list(rel_properties)
#             self.entity_types[entity_type_name].reverse_relationships = list(rev_rel_properties)
#     def __str__(self) -> str:
#         return """
# Schema(
#     Meta properties:
#         {}
#         {}
#     Data properties: 
#         {}
#     Relationships:
#         {}
#     Entity types:
#         {}
# )""".format(self.id_property,
#             self.entity_type_property,
#             "\n        ".join([str(f) for f in self.properties.values()]),
#             "\n        ".join([str(f) for f in self.relationships.values()]),
#             "\n        ".join([str(et) for et in self.entity_types.values()])
# )

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
        with open(conf_file, "rt") as ifd:
            config_rules += json.loads(ifd.read())

    expanded_schema = expand(schema, config_rules)
    expanded_schema["entity_type_property"] = schema["meta"]["entity_type_property"]
    expanded_schema["id_property"] = schema["meta"]["id_property"]

    with open(args.output, "wt") as ofd:
        ofd.write(json.dumps(expanded_schema, indent=2))
