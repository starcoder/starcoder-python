from starcoder.entity import EntityType, UnpackedEntity, PackedEntity
from starcoder.field import Field, DataField, RelationshipField, EntityTypeField, IdField, PackedValueType, UnpackedValueType
from typing import List, Dict, Any, Set, cast, MutableSequence, Type
import logging

logger = logging.getLogger(__name__)

class Schema(object):
    """
A Schema object represents and tracks entity-relationship information.

An instance should be treated as, essentially, three dictionaries
which map from strings (names) to objects:

  entity_types, data_fields, relationship_fields

and two properties:

  id_field, entity_type_field

Schema objects also store the original JSON object they were
constructed from in the "json" property.
    """
    def __init__(self, spec: Dict[str, Any], field_classes: Dict[str, Type[Field]]) -> None: # this "Any" is OK
        self.json = spec.copy()
        self.id_field: IdField = cast(IdField, field_classes["id"](spec["meta"]["id_field"]))
        self.entity_type_field: EntityTypeField = cast(EntityTypeField, field_classes["entity_type"](spec["meta"]["entity_type_field"]))
        self.data_fields: Dict[str, DataField[Any, Any, Any]] = {} # these "Any"s are OK
        self.relationship_fields: Dict[str, RelationshipField] = {}
        self.entity_types = {}
        self.seen_entity_types: Set[UnpackedValueType] = set()
        for field_name, field_spec in spec["data_fields"].items():
            if field_spec.get("ignore", False) == False:
                field_type = field_spec["type"]
                self.data_fields[field_name] = cast(DataField[Any, Any, Any], field_classes[field_type](field_name, **field_spec)) # these "Any"s are OK
        for field_name, field_spec in spec["relationship_fields"].items():
            if field_spec.get("ignore", False) == False:
                self.relationship_fields[field_name] = cast(RelationshipField, field_classes["relationship"](field_name, type="relationship", **field_spec))
        for entity_type, entity_spec in spec["entity_types"].items():
            if entity_spec.get("ignore", False) == False:
                rel_fields = set([k for k, v in self.relationship_fields.items() if v.source_entity_type == entity_type])
                rev_rel_fields = set([k for k, v in self.relationship_fields.items() if v.target_entity_type == entity_type])
                data_fields = set([k for k in entity_spec["data_fields"] if k in self.data_fields])
                self.entity_types[entity_type] = EntityType(entity_type,
                                                            data_fields,
                                                            rel_fields,
                                                            rev_rel_fields)

    def pack(self, entity: UnpackedEntity) -> PackedEntity:
        entity = {k : entity.get(k, None) for k in self.all_field_names} # type: ignore
        retval = {k : self.data_fields[k].pack(v) if k in self.data_fields else v for k, v in entity.items()}
        return retval

    def unpack(self, entity: PackedEntity) -> UnpackedEntity:
        retval = {k : self.data_fields[k].unpack(v) if k in self.data_fields else v for k, v in entity.items()}
        #reveal_type(retval)
                  #if k in self.data_fields for k, v in entity.items()} # else v for k, v in entity.items()}
        return {k : v for k, v in retval.items() if v not in [[], None]} # type: ignore

    @property
    def all_field_names(self) -> List[str]:
        return list(self.data_fields.keys()) + list(self.relationship_fields.keys()) + [self.id_field.name, self.entity_type_field.name]

    @property    
    def all_field_objects(self) -> List[Field]:
       return cast(List[Field], list(self.data_fields.values())) + cast(List[Field], list(self.relationship_fields.values())) + cast(List[Field], [self.id_field, self.entity_type_field])
    
    def observe_entity(self, entity: UnpackedEntity) -> None:
        if entity[self.entity_type_field.name] in self.entity_types:
            self.seen_entity_types.add(entity[self.entity_type_field.name])
            for k, v in entity.items():
                if k in self.data_fields:
                    self.data_fields[k].observe_value(v)

    def verify(self) -> bool:
        return True
        #raise Exception("Field '{}' had no observed values".format(name))

    def minimize(self) -> None:
        #print(self.seen_entity_types)
        # remove fields that haven't been seen
        for field_name in list(self.data_fields.keys()):
            if self.data_fields[field_name].empty == True:
                #print(field_name)
                del self.data_fields[field_name]
        for entity_type_name in list(self.entity_types.keys()):
            # remove undefined fields from entity-types
            self.entity_types[entity_type_name].data_fields = [f for f in self.entity_types[entity_type_name].data_fields if f in self.data_fields]
            # remove entity-types that haven't been seen
            if entity_type_name not in self.seen_entity_types:
                del self.entity_types[entity_type_name]

        # remove fields that don't occur in an entity-type
        for field_name in list(self.data_fields.keys()):
            if all([field_name not in e.data_fields for e in self.entity_types.values()]):
                #print(field_name)
                del self.data_fields[field_name]
        # remove relationships for which an entity-type doesn't exist
        for field_name in list(self.relationship_fields.keys()):
            rel_field = self.relationship_fields[field_name]
            if rel_field.source_entity_type not in self.entity_types or rel_field.target_entity_type not in self.entity_types:
                del self.relationship_fields[field_name]
        for entity_type_name in list(self.entity_types.keys()):
            rel_fields = set([k for k, v in self.relationship_fields.items() if v.source_entity_type == entity_type_name])
            rev_rel_fields = set([k for k, v in self.relationship_fields.items() if v.target_entity_type == entity_type_name])
            self.entity_types[entity_type_name].relationship_fields = list(rel_fields)
            self.entity_types[entity_type_name].reverse_relationship_fields = list(rev_rel_fields)
    def __str__(self) -> str:
        return """
Schema(
    Meta fields:
        {}
        {}
    Data fields: 
        {}
    Relationship fields:
        {}
    Entity types:
        {}
)""".format(self.id_field,
            self.entity_type_field,
            "\n        ".join([str(f) for f in self.data_fields.values()]),
            "\n        ".join([str(f) for f in self.relationship_fields.values()]),
            "\n        ".join([str(et) for et in self.entity_types.values()])
)
                
if __name__ == "__main__":
    pass
