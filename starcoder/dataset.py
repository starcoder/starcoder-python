import pickle
import re
import argparse
import json
import gzip
import functools

import logging
import numpy
import scipy.sparse
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse.csgraph import connected_components
import torch
import math
import uuid
from starcoder.random import random
from starcoder.entity import UnpackedEntity, PackedEntity, ID, Index
from starcoder.field import UnpackedValueType, PackedValueType
from starcoder.adjacency import Adjacency, Adjacencies
from starcoder.schema import Schema
from typing import List, Dict, Any, Tuple, Sequence, Hashable, MutableSequence, NewType, cast

logger = logging.getLogger(__name__)

class Dataset(torch.utils.data.Dataset): # type: ignore
    """
    The Dataset class is needed mainly for operations that depend on
    graph structure, particularly those that require connected components.
    A Dataset is, basically, a Schema and a list of JSON objects.
    """
    def __init__(self, schema: Schema, entities: MutableSequence[Dict[str, Any]], strict: bool=False):
        super(Dataset, self).__init__()
        self.schema = schema
        known_field_names = schema.all_field_names
        self.entities = []
        self.id_to_index: Dict[ID, Index] = {}
        self.index_to_id: Dict[Index, ID] = {}
        for idx, entity in [(cast(Index, x), y) for x, y in enumerate(entities)]:
            entity_id = cast(ID, entity[self.schema.id_field.name])
            if entity_id in self.id_to_index:
                raise Exception("Entity with id {} already exists".format(entity_id))
            #assert entity_id not in self.id_to_index
            #assert idx not in self.index_to_id
            self.id_to_index[entity_id] = idx
            self.index_to_id[idx] = entity_id #[self.schema.id_field.name]
            for k in entity.keys():
                if k not in known_field_names and strict==True:
                    raise Exception("Unknown field: '{}'".format(k))
            self.entities.append(entity)
        self.edges: Dict[str, Dict[int, MutableSequence[int]]] = {}
        for entity in self.entities:
            entity_type = entity[self.schema.entity_type_field.name]
            if entity_type not in self.schema.entity_types:
                continue
            entity_id = entity[self.schema.id_field.name]
            source_index = self.id_to_index[entity_id]            
            for relationship_field in self.schema.entity_types[entity_type].relationship_fields:
                target_ids = entity.get(relationship_field, [])
                for target in target_ids if isinstance(target_ids, list) else [target_ids]:
                    if target not in self.id_to_index:
                        logger.debug("Could not find target %s for entity %s relationship %s", target, entity_id, relationship_field)
                        continue
                    target_index = self.id_to_index[target]
                    self.edges[relationship_field] = self.edges.get(relationship_field, {})
                    self.edges[relationship_field][source_index] = self.edges[relationship_field].get(source_index, [])
                    self.edges[relationship_field][source_index].append(target_index)
        self.update_components()
        
    def get_type_ids(self, *type_names: Sequence[str]) -> List[ID]:
        retval = []
        for i in self.ids:
            if self.entity(i)[self.schema.entity_type_field.name] in type_names:
                retval.append(i)
        return retval

    def subselect_entities_by_id(self, ids: Sequence[ID], invert: bool=False) -> "Dataset":
        if invert:
            data = Dataset(self.schema, [self.entities[i] for i in range(len(self)) if self.index_to_id[cast(Index, i)] not in ids])
        else:
            data = Dataset(self.schema, [self.entities[self.id_to_index[eid]] for eid in ids])
        return data
    
    def subselect_entities(self, ids: Sequence[ID], invert: bool=False) -> "Dataset":
        return self.subselect_entities_by_id(ids, invert)

    def subselect_components(self, indices: Sequence[int]) -> "Dataset":
        return Dataset(self.schema, [self.entities[self.id_to_index[eid]] for eid in sum([self.components[i][0] for i in indices], [])])

    @property
    def ids(self) -> List[ID]:
        return [i for i in self.id_to_index.keys()]

    @property
    def aggregate_adjacencies(self) -> Adjacency:
        rows, cols, vals = [], [], []
        for _, rs in self.edges.items():
            for r, cs in rs.items():
                for c in cs:
                    rows.append(r)
                    cols.append(c)
                    vals.append(True)
        adjacency = scipy.sparse.csr_matrix((vals, (rows, cols)), 
                                            shape=(self.num_entities, self.num_entities), 
                                            dtype=numpy.bool)
        return adjacency
        
    def update_components(self) -> None:
        # # create union adjacency matrix
        # rows, cols, vals = [], [], []
        # for _, rs in self.edges.items():
        #     for r, cs in rs.items():
        #         for c in cs:
        #             rows.append(r)
        #             cols.append(c)
        #             vals.append(True)
        # adjacency = scipy.sparse.csr_matrix((vals, (rows, cols)), 
        #                                     shape=(self.num_entities, self.num_entities), 
        #                                     dtype=numpy.bool)
        adjacency = self.aggregate_adjacencies
        # create list of connected components
        num, ids = connected_components(adjacency)
        components_to_indices: Dict[int, MutableSequence[Index]] = {}    
        for i, c in enumerate(ids):
            components_to_indices[c] = components_to_indices.get(c, [])
            components_to_indices[c].append(cast(Index, i))
            
        #largest_component_size = 0 if len(components) == 0 else max([len(x) for x in components.values()])
        if len(components_to_indices) == 0:
            raise Exception("The data is empty: this probably isn't what you want.  Perhaps add more instances, or adjust the train/dev/test split proportions?")
        
        self.components: MutableSequence[Tuple[List[ID], Adjacencies]] = []
        #reveal_type(components_to_indices)
        for indices in components_to_indices.values():
            #reveal_type(indices)
            #component_adjacencies = {} #{k : numpy.full((len(c), len(c)), False) for k in self.edges.keys()}
            ca_rows: Dict[Any, Any] = {}
            ca_cols: Dict[Any, Any] = {}
            g2l = {k : v for v, k in enumerate(indices)}
            for gsi in indices:
                lsi = g2l[gsi]
                for rel_type, rows in self.edges.items():
                    ca_rows[rel_type] = ca_rows.get(rel_type, [])
                    ca_cols[rel_type] = ca_cols.get(rel_type, [])
                    for gti in rows.get(gsi, []):
                        lti = g2l[cast(Index, gti)]
                        ca_rows[rel_type].append(lsi)
                        ca_cols[rel_type].append(lti)
                        #component_adjacencies[rel_type][lsi, lti] = True
            component_adjacencies = {rel_type : scipy.sparse.csr_matrix(([True for _ in ca_rows[rel_type]], (ca_rows[rel_type], ca_cols[rel_type])), 
                                                                 shape=(len(indices), len(indices)), dtype=numpy.bool) for rel_type in self.edges.keys()}
            self.components.append(([self.index_to_id[c] for c in indices], component_adjacencies)) # = [c for c in components.values()]

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def num_entities(self) -> int:
        return len(self.entities)
    
    def component_ids(self, i: int) -> List[ID]:
        return self.components[i][0]

    def component_adjacencies(self, i: int) -> Adjacencies:
        return self.components[i][1]

    def entity(self, i: ID) -> UnpackedEntity:
        return self.entities[self.id_to_index[i]]
    
    def component(self, i: int) -> Tuple[List[UnpackedEntity], Adjacencies]:
        entity_ids, adjacencies = self.components[i]

        #entities = [self.entity(i) for i in entity_indices]
        #assert all([len(entities) == v.shape[0] for v in adjacencies.values()])
        return ([self.entity(i) for i in entity_ids], adjacencies.copy())

    #@property
    #def num_components(self) -> int:
    #    return len(self.components)

    def __str__(self) -> str:
        return "Dataset({} entities, {} components)".format(self.num_entities,
                                                            self.num_components)

if __name__ == "__main__":
    pass
