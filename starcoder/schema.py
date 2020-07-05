from starcoder.registry import field_classes
import logging

logger = logging.getLogger(__name__)

class EntityType(object):
    def __init__(self, name, data_fields, relation_fields, reverse_relation_fields):
        self.name = name
        self.data_fields = list(sorted(data_fields))
        self.relation_fields = list(sorted(relation_fields))
        self.reverse_relation_fields = list(sorted(reverse_relation_fields))
    def __str__(self):
        return "{}: data fields={}, relation fields={}".format(self.name, self.data_fields, self.relation_fields)

class EncodedEntity(dict):
    def __init__(self, *argv, **argd):
        super(EncodedEntity, self).__init__(*argv, **argd)

class DecodedEntity(dict):
    def __init__(self, *argv, **argd):
        super(DecodedEntity, self).__init__(*argv, **argd)
        
class Schema(object):
    """
A Schema object represents and tracks entity-relation information.

An instance should be treated as, essentially, three dictionaries
which map from strings (names) to objects:

  entity_types, data_fields, relation_fields

and two properties:

  id_field, entity_type_field

Schema objects also store the original JSON object they were
constructed from in the "json" property.
    """
    def __init__(self, spec):
        self.json = spec.copy()
        self.id_field = field_classes["id"](spec["meta"]["id_field"])
        self.entity_type_field = field_classes["entity_type"](spec["meta"]["entity_type_field"])
        self.data_fields = {}
        self.relation_fields = {}
        self.entity_types = {}

        for field_name, field_spec in spec["data_fields"].items():
            if field_spec.get("ignore", False) == False:
                field_type = field_spec["type"]
                self.data_fields[field_name] = field_classes[field_type](field_name, **field_spec)
        for field_name, field_spec in spec["relation_fields"].items():
            if field_spec.get("ignore", False) == False:
                self.relation_fields[field_name] = field_classes["relation"](field_name, type="relation", **field_spec)
        for entity_type, entity_spec in spec["entity_types"].items():
            if entity_spec.get("ignore", False) == False:
                rel_fields = set([k for k, v in self.relation_fields.items() if v.source_entity_type == entity_type])
                rev_rel_fields = set([k for k, v in self.relation_fields.items() if v.target_entity_type == entity_type])
                data_fields = set([k for k in entity_spec["data_fields"] if k in self.data_fields])
                self.entity_types[entity_type] = EntityType(entity_type,
                                                            data_fields,
                                                            rel_fields,
                                                            rev_rel_fields)

    def encode(self, entity: DecodedEntity) -> EncodedEntity:
        assert isinstance(entity, DecodedEntity)
        retval = EncodedEntity({k : self.data_fields[k].encode(v) if k in self.data_fields else v for k, v in entity.items()})
        return retval

    def decode(self, entity: EncodedEntity) -> DecodedEntity:
        assert isinstance(entity, EncodedEntity)
        retval = DecodedEntity({k : self.data_fields[k].decode(v) if k in self.data_fields else v for k, v in entity.items()})
        return retval
    
    def observe_entity(self, entity):
        for k, v in entity.items():
            if k in self.data_fields:
                self.data_fields[k].observe_value(v)
            
    def __str__(self):
        return """
Schema(
    Fields:
        {}
        {} 
        {}
        {}
    entity types:
        {}
)
""".format(self.id_field,
           self.entity_type_field,
           "\n        ".join([str(f) for f in self.data_fields.values()]),
           "\n        ".join([str(f) for f in self.relation_fields.values()]),
           "\n        ".join([str(et) for et in self.entity_types.values()])
)
                
if __name__ == "__main__":
    pass
