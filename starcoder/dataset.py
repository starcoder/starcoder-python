import pickle
import re
import sys
import argparse
import json
import gzip
import functools
import random
import logging
import numpy
import scipy.sparse
from sklearn.metrics import f1_score, accuracy_score
from scipy.sparse.csgraph import connected_components
import torch
import math
import uuid
from starcoder import fields
from starcoder.schema import EncodedEntity, DecodedEntity

logger = logging.getLogger(__name__)

class Dataset(object):
    """
    The Dataset class is needed mainly for operations that depend on
    graph structure, particularly those that require connected components.
    A Dataset is, basically, a Schema and a list of JSON objects.
    """
    def __init__(self, schema, entities):
        self.schema = schema
        self._entities = []
        self._entity_fields = {}
        self.id_to_index = {}
        self.index_to_id = {}
        for idx, entity in enumerate(entities):
            #entity_type = entity[self._spec.entity_type_field]
            #entity_id = entity[self._spec.id_field]
            #assert (entity_id not in self._id_to_index), "The id field ('{}') must be unique".format(self._spec.id_field)
            self.id_to_index[entity[self.schema.id_field.name]] = idx
            self.index_to_id[idx] = entity[self.schema.id_field.name]
            #self._entity_fields[entity_type] = self._spec.entity_fields(entity_type) #
            self._entities.append(DecodedEntity(entity))
        self._edges = {}
        for entity in self._entities:
            entity_type = entity[self.schema.entity_type_field.name]
            if entity_type not in self.schema.entity_types:
                continue
            entity_id = entity[self.schema.id_field.name]
            source_index = self.id_to_index[entity_id]            
            for relation_field in self.schema.entity_types[entity_type].relation_fields:
                target_ids = entity.get(relation_field, [])
                for target in target_ids if isinstance(target_ids, list) else [target_ids]:
                    if target not in self.id_to_index:
                        logger.debug("Could not find target %s for entity %s relation %s", target, entity_id, relation_field)
                        continue
                    target_index = self.id_to_index[target]
                    self._edges[relation_field] = self._edges.get(relation_field, {})
                    self._edges[relation_field][source_index] = self._edges[relation_field].get(source_index, [])
                    self._edges[relation_field][source_index].append(target_index)
        self._update_components()

    def get_type_indices(self, *type_names):
        retval = []
        for i in range(len(self)):
            if self._entities[i][self.schema.entity_type_field.name] in type_names:
                retval.append(i)
        return retval
    
    def subselect_entities_by_index(self, indices, invert=False):
        if invert:
            data = Dataset(self.schema, [self._entities[i] for i in range(len(self)) if i not in indices])
        else:
            data = Dataset(self.schema, [self._entities[i] for i in indices])
        return data

    def subselect_entities_by_id(self, ids, invert=False):
        if invert:
            data = Dataset(self.schema, [self._entities[i] for i in range(len(self)) if self.index_to_id[i] not in ids])
        else:
            data = Dataset(self.schema, [self._entities[self.id_to_index[i]] for i in ids])
        return data    

    def subselect_components(self, indices):
        return Dataset(self.schema, [self._entities[j] for j in sum([self._components[i][0] for i in indices], [])])
    
    def encode(self, item):
        return self.schema.encode(item)

    def decode(self, item):
        return self.schema.decode(item)

    def _update_components(self):
        # create union adjacency matrix
        rows, cols, vals = [], [], []
        for _, rs in self._edges.items():
            for r, cs in rs.items():
                for c in cs:
                    rows.append(r)
                    cols.append(c)
                    vals.append(True)
        adjacency = scipy.sparse.csr_matrix((vals, (rows, cols)), 
                                            shape=(len(self), len(self)), 
                                            dtype=numpy.bool)

        # create list of connected components
        num, ids = connected_components(adjacency)
        components = {}    
        for i, c in enumerate(ids):
            components[c] = components.get(c, [])
            components[c].append(i)
            
        largest_component_size = 0 if len(components) == 0 else max([len(x) for x in components.values()])
        if len(components) == 0:
            raise Exception("The data is empty: this probably isn't what you want.  Perhaps add more instances, or adjust the train/dev/test split proportions?")
        
        self._components = []
        for c in components.values():
            #component_adjacencies = {} #{k : numpy.full((len(c), len(c)), False) for k in self._edges.keys()}
            ca_rows, ca_cols = {}, {}
            g2l = {k : v for v, k in enumerate(c)}
            for gsi in c:
                lsi = g2l[gsi]
                for rel_type, rows in self._edges.items():
                    ca_rows[rel_type] = ca_rows.get(rel_type, [])
                    ca_cols[rel_type] = ca_cols.get(rel_type, [])
                    for gti in rows.get(gsi, []):                        
                        lti = g2l[gti]
                        ca_rows[rel_type].append(lsi)
                        ca_cols[rel_type].append(lti)
                        #component_adjacencies[rel_type][lsi, lti] = True
            component_adjacencies = {rel_type : scipy.sparse.csr_matrix(([True for _ in ca_rows[rel_type]], (ca_rows[rel_type], ca_cols[rel_type])), 
                                                                 shape=(len(c), len(c)), dtype=numpy.bool) for rel_type in self._edges.keys()}
            #assert all([len(c) == v.shape[0] for v in component_adjacencies.values()])
            self._components.append((c, component_adjacencies)) # = [c for c in components.values()]

        
    def __getitem__(self, index):
        return self._entities[index]
    
    def __len__(self):
        return len(self._entities)

    def component_indices(self, i):
        return self._components[i][0]

    def component_adjacencies(self, i):
        return self._components[i][1]
    
    def component(self, i):
        entity_indices, adjacencies = self._components[i]
        entities = [self[i] for i in entity_indices]
        #assert all([len(entities) == v.shape[0] for v in adjacencies.values()])
        return (entities, adjacencies.copy())

    @property
    def num_components(self):
        return len(self._components)

    def __str__(self):
        return "Dataset({} entities, {} components with max size {})".format(len(self),
                                                                             len(self._components),
                                                                             0 if len(self._components) == 0 else max([len(c) for c, _ in self._components]))


if __name__ == "__main__":
    pass
