import logging
import torch
import numpy
from starcoder.property import Property, PackedValueType, UnpackedValueType
from torch import Tensor, dtype
from typing import List, Dict, Any, Set, Union, NewType

logger = logging.getLogger(__name__)

class EntityType(object):
    """
    Fully describes a sort of "thing" in terms of its potential properties and relationships to other "things"
    """
    def __init__(self,
                 name: str,
                 properties: Union[Set[str], List[str]],
                 relationships: Union[Set[str], List[str]],
                 reverse_relationships: Union[Set[str], List[str]]) -> None:
        self.name = name
        self.properties = list(sorted(properties))
        self.relationships = list(sorted(relationships))
        self.reverse_relationships = list(sorted(reverse_relationships))
    def __str__(self) -> str:
        return "{}: data={}, relationships={}".format(self.name, self.properties, self.relationships)


    
# A human-friendly entity, where property values have a transparent, natural meaning, e.g. as read from a JSON file.
UnpackedEntity = Dict[str, UnpackedValueType]
UnpackedEntities = List[UnpackedEntity]

# A computer-friendly entity, where property-values are all numeric, and often not directly interpretable.  See
# the pack/unpack methods on Schema and Property objects for conversion.
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
        #property_names = set(sum([[k for k in e.keys()] for e in encoded_entities], [])) #list(schema.properties.keys())))
        #self.entities = {}
        #for entity in packed_entities:
        #    self.entities[entity
        
        full_entities: Dict[str, List[Any]] = {k : [] for k in property_names} #{k : [] for k in property_names}
        for entity in encoded_entities:
            for property_name in property_names:
                if property_name in properties:
                    full_entities[property_name].append(entity.get(property_name, properties[property_name].missing_value))
                else:
                    full_entities[property_name].append(entity.get(property_name, False))
        return {k : numpy.array(v) if k not in properties else torch.tensor(v, dtype=properties[k].stacked_type) for k, v in full_entities.items()}
        
        pass



Index = NewType("Index", int)
ID = NewType("ID", str)

def stack_entities(encoded_entities, properties): # -> StackedEntities:
    """
    Converts a list of packed entities into pandas/R-like format ("stacked"), i.e. a dictionary of properties 
    where each property has a tensor/multi-dimensional array of numeric values equal in first dimension to 
    the number of entities, with float("nan") for missing items.    
    """
    #print(encoded_entities)
    #sys.exit()
    property_names = set(sum([[k for k in e.keys()] for e in encoded_entities], [])) #list(schema.properties.keys())))
    full_entities: Dict[str, List[Any]] = {k : [] for k in property_names} #{k : [] for k in property_names}
    for entity in encoded_entities:
        for property_name in property_names:
            if property_name in properties:                
                full_entities[property_name].append(entity.get(property_name, properties[property_name].missing_value))    
            else:
                full_entities[property_name].append(entity.get(property_name, False))
    #for k, v in full_entities.items():
    #    if k in properties:
    #        print(k, type(v))
    #        print(v)
    #        torch.tensor(v, dtype=properties[k].stacked_type)

    # for k, v in full_entities.items():
    #     if k in properties:
    #         print(k)
    #         print(v)
    #         y = torch.tensor(v, dtype=properties[k].stacked_type)
    return {k : numpy.array(v) if k not in properties else torch.tensor(v, dtype=properties[k].stacked_type) for k, v in full_entities.items()}

def unstack_entities(stacked_entities: StackedEntities, schema) -> PackedEntities: #properties: Dict[str, Property[Any, Any, Any]]) -> PackedEntities:
    """
    Converts pandas/R-like ("stacked") data into a list of entities, where a missing property simply won't
    have an entry in the given entity.
    """
    property_names = list(stacked_entities.keys())
    retval = []
    for i in range(stacked_entities[property_names[0]].shape[0]):
        entity_type = stacked_entities[schema.entity_type_property.name][i]
        entity_id = stacked_entities[schema.id_property.name][i]
        entity = {schema.entity_type_property.name : entity_type, schema.id_property.name : entity_id}
        for property_name in schema.entity_types[entity_type].properties:
            if stacked_entities[property_name][i] != schema.properties[property_name].missing_value:
                entity[property_name] = stacked_entities[property_name][i].tolist()
            elif stacked_entities[property_name][i] != None:
                entity[property_name] = stacked_entities[property_name][i].tolist()
        retval.append(entity)
    return retval
