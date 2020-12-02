import logging
import torch
import numpy
from starcoder.field import Field, DataField, PackedValueType, UnpackedValueType
from torch import Tensor, dtype
from typing import List, Dict, Any, Set, Union, NewType

logger = logging.getLogger(__name__)

class EntityType(object):
    """
    Fully describes a sort of "thing" in terms of its potential fields and relationships to other "things"
    """
    def __init__(self,
                 name: str,
                 data_fields: Union[Set[str], List[str]],
                 relationship_fields: Union[Set[str], List[str]],
                 reverse_relationship_fields: Union[Set[str], List[str]]) -> None:
        self.name = name
        self.data_fields = list(sorted(data_fields))
        self.relationship_fields = list(sorted(relationship_fields))
        self.reverse_relationship_fields = list(sorted(reverse_relationship_fields))
    def __str__(self) -> str:
        return "{}: data={}, relationships={}".format(self.name, self.data_fields, self.relationship_fields)


    
# A human-friendly entity, where field values have a transparent, natural meaning, e.g. as read from a JSON file.
UnpackedEntity = Dict[str, UnpackedValueType]
UnpackedEntities = List[UnpackedEntity]

# A computer-friendly entity, where field-values are all numeric, and often not directly interpretable.  See
# the pack/unpack methods on Schema and DataField objects for conversion.
PackedEntity = Dict[str, Union[PackedValueType,
                                List[PackedValueType],
                                List[List[PackedValueType]],
                                List[List[List[PackedValueType]]],
                                List[List[List[List[PackedValueType]]]]]]
PackedEntities = List[PackedEntity]

# packed entities, but arranged in pandas/R-style
StackedEntities = Dict[str, Union[Tensor, numpy.array]]
class fStackedEntities(object):
    def __init__(self, packed_entities: PackedEntities):
        #field_names = set(sum([[k for k in e.keys()] for e in encoded_entities], [])) #list(schema.data_fields.keys())))
        #self.entities = {}
        #for entity in packed_entities:
        #    self.entities[entity
        
        full_entities: Dict[str, List[Any]] = {k : [] for k in field_names} #{k : [] for k in field_names}
        for entity in encoded_entities:
            for field_name in field_names:
                if field_name in data_fields:
                    full_entities[field_name].append(entity.get(field_name, data_fields[field_name].missing_value))
                else:
                    full_entities[field_name].append(entity.get(field_name, False))
        return {k : numpy.array(v) if k not in data_fields else torch.tensor(v, dtype=data_fields[k].stacked_type) for k, v in full_entities.items()}
        
        pass



Index = NewType("Index", int)
ID = NewType("ID", str)

def stack_entities(encoded_entities: PackedEntities, data_fields: Dict[str, DataField[Any, Any, Any]]) -> StackedEntities:
    """
    Converts a list of packed entities into pandas/R-like format ("stacked"), i.e. a dictionary of fields 
    where each field has a tensor/multi-dimensional array of numeric values equal in first dimension to 
    the number of entities, with float("nan") for missing items.    
    """
    #print(encoded_entities)
    #sys.exit()
    field_names = set(sum([[k for k in e.keys()] for e in encoded_entities], [])) #list(schema.data_fields.keys())))
    full_entities: Dict[str, List[Any]] = {k : [] for k in field_names} #{k : [] for k in field_names}
    for entity in encoded_entities:
        for field_name in field_names:
            if field_name in data_fields:                
                full_entities[field_name].append(entity.get(field_name, data_fields[field_name].missing_value))    
            else:
                full_entities[field_name].append(entity.get(field_name, False))

    return {k : numpy.array(v) if k not in data_fields else torch.tensor(v, dtype=data_fields[k].stacked_type) for k, v in full_entities.items()}

def unstack_entities(stacked_entities: StackedEntities, schema) -> PackedEntities: #data_fields: Dict[str, DataField[Any, Any, Any]]) -> PackedEntities:
    """
    Converts pandas/R-like ("stacked") data into a list of entities, where a missing field simply won't
    have an entry in the given entity.
    """
    field_names = list(stacked_entities.keys())
    retval = []
    for i in range(stacked_entities[field_names[0]].shape[0]):
        entity_type = stacked_entities[schema.entity_type_field.name][i]
        entity_id = stacked_entities[schema.id_field.name][i]
        entity = {schema.entity_type_field.name : entity_type, schema.id_field.name : entity_id}
        for field_name in schema.entity_types[entity_type].data_fields:
            if stacked_entities[field_name][i] != schema.data_fields[field_name].missing_value:
                entity[field_name] = stacked_entities[field_name][i].tolist()
            elif stacked_entities[field_name][i] != None:
                entity[field_name] = stacked_entities[field_name][i].tolist()
        retval.append(entity)
    return retval
