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




class EntityTypeField(dict):

    name = "entity_type"

    def __init__(self, field_values):
        self[Missing] = Missing.value
        self[Unknown] = Unknown.value
        for value in field_values:
            self[value] = self.get(value, len(self))
        self._rlookup = {v : k for k, v in self.items()}

    def __str__(self):
        return "EntityType(possible_values={})".format(len(self))

    def encode(self, v):
        return self.get(v, self[Unknown])

    def decode(self, v):
        if v not in self._rlookup:
            raise Exception("Could not decode value '{}' (lookup={}, type={})".format(v, self._rlookup, type(v)))
        return self._rlookup[v]



class RelationField(object):
    name = "relation"
    def __init__(self, field_values):
        self._source, self._target = field_values
    def encode(self, v):
        return v
    def decode(self, v):
        return v

class IdField(object):
    name = "id"
    def __init__(self, field_values):
        pass

    def encode(self, v):
        return v

    def decode(self, v):
        return v


class NumericField(object):

    name = "numeric"

    def __init__(self, field_values):
        field_values = [float(x) for x in field_values]
        self._minimum = min(field_values)
        self._maximum = max(field_values)
        self._trivial = self._minimum == self._maximum

    def __str__(self):
        return "Numeric(min/max={}/{})".format(self._minimum, self._maximum)

    def encode(self, v):
        return (0.0 if self._trivial else (v - self._minimum) / (self._maximum - self._minimum))

    def decode(self, v):
        return (self._minimum if self._trivial else (v * (self._maximum - self._minimum) + self._minimum))


class DistributionField(object):

    name = "distribution"

    def __init__(self, field_values):
        self._categories = list(sorted(set(field_values)))

    def __str__(self):
        return "Distribution({})".format(len(self._categories))

    def encode(self, v):
        total = sum(v.values())
        return [0.0 if c not in v else (v[c] / total) for c in self._categories]

    def decode(self, v):
        assert(len(v) == len(self._categories))
        retval = {}
        if all([x >= 0 for x in v]):
            total = sum([x for x in v])
            for k, p in zip(self._categories, v):
                if p > 0:
                    retval[k] = p / total
        elif all([x <= 0 for x in v]):            
            total = sum([math.exp(x) for x in v])
            for k, p in zip(self._categories, v):
                retval[k] = math.exp(p) / total            
        else:
            raise Exception("Got probabilities that were not all of the same sign!")
        return retval


class IntegerField(object):

    name = "numeric"

    def __init__(self, field_values):
        field_values = [float(x) for x in field_values]
        self._minimum = min(field_values)
        self._maximum = max(field_values)
        self._trivial = self._minimum == self._maximum

    def __str__(self):
        return "Integer(min/max={}/{})".format(self._minimum, self._maximum)

    def encode(self, v):
        return (0.0 if self._trivial else (v - self._minimum) / (self._maximum - self._minimum))

    def decode(self, v):
        return round(self._minimum if self._trivial else (v * (self._maximum - self._minimum) + self._minimum))


class Missing(object):
    value = 0


class Unknown(object):
    value = 1
    

class CategoricalField(dict):

    name = "categorical"

    def __init__(self, field_values):
        self[Missing] = Missing.value
        self[Unknown] = Unknown.value
        for value in field_values:
            self[value] = self.get(value, len(self))
        self._rlookup = {v : k for k, v in self.items()}

    def __str__(self):
        return "Categorical(possible_values={})".format(len(self))

    def encode(self, v):
        return self.get(v, self[Unknown])

    def decode(self, v):
        if v not in self._rlookup:
            raise Exception("Could not decode value '{}' (lookup={}, type={})".format(v, self._rlookup, type(v)))
        return self._rlookup[v]


class SequentialField(dict):

    name = "sequential"

    def __init__(self, field_values):
        unique_sequences = set()
        self[Missing] = Missing.value
        self[Unknown] = Unknown.value
        self._max_length = 0
        for value in field_values:
            unique_sequences.add(value)
            self._max_length = max(self._max_length, len(value))
            for element in value:
                self[element] = self.get(element, len(self))
        self._rlookup = {v : k for k, v in self.items()}
        self._unique_sequence_count = len(unique_sequences)

    def __str__(self):
        return "Sequential(unique_elems={}, unique_seqs={}, max_length={})".format(len(self),
                                                                                   self._unique_sequence_count, 
                                                                                   self._max_length)

    def encode(self, v):
        return [self.get(e, self[Unknown]) for e in v]

    def decode(self, v):
        try:
            return "".join([self._rlookup[e] for e in v if e not in [Missing.value, Unknown.value]])
        except:
            raise Exception("Could not decode values '{}' (lookup={}, type={})".format(v, self._rlookup, type(v[0])))


field_classes = {"numeric" : NumericField,
                 "categorical" : CategoricalField,
                 "sequential" : SequentialField,
                 "integer" : IntegerField,
                 "keyword" : CategoricalField,
                 "text" : SequentialField,
                 "relation" : RelationField,
                 "distribution" : DistributionField,
                 "id" : IdField,
                 "entity_type" : EntityTypeField,
             }


class Datum(dict):
    def __init__(self, spec, obj):
        self._identifier = uuid.uuid4() if identifier == None else identifier
        self._unknown_fields = {}
        for k, v in obj.items():
            pass


class Spec(object):

    def __init__(self, field_name_to_type, field_name_to_values, entity_type_to_field_names, symmetric=False):
        self._entity_type_to_field_names = entity_type_to_field_names
        self._field_name_to_object = {}
        self._entity_type_to_relation_types = {}
        self._id_field = None
        for field_name, field_type in field_name_to_type.items():
            if field_type not in field_classes:
                raise Exception("No class corresponding to field type '{}'".format(field_type))
            if field_type == "id":
                self._id_field = field_name
            elif field_type == "entity_type":
                self._entity_type_field = field_name
            self._field_name_to_object[field_name] = field_classes[field_type](field_name_to_values.get(field_name, set()))

    def __str__(self):
        return "Spec(entities: {}, fields: {}, id_field: '{}', entity_type_field: '{}')".format(self.entity_types,
                                                                                                self.field_names,
                                                                                                self._id_field,
                                                                                                self._entity_type_field)

    def entity_relations(self, entity_type):
        return set([x for x in self.entity_fields(entity_type) if isinstance(self.field_object(x), RelationField)])

    @property
    def regular_field_names(self):
        return set([f for f, o in self._field_name_to_object.items() if not isinstance(o, (IdField, RelationField))])

    @property
    def entity_types(self):
        return set(self._entity_type_to_field_names.keys())

    def entity_fields(self, entity_name):
        return self._entity_type_to_field_names.get(entity_name, set())

    def field_object(self, field_name):
        return self._field_name_to_object[field_name]

    def exists(self, field_name):
        return (field_name in self._field_name_to_object)

    @property
    def field_names(self):
        return set([f for f in self._field_name_to_object.keys()])

    @property
    def entity_type_field(self):
        return self._entity_type_field

    @property
    def id_field(self):
        return self._id_field

    @property
    def has_id_field(self):
        return (self._id_field != None)

    @property
    def num_fields(self):
        return len(self._field_name_to_object)

    @property
    def num_entities(self):
        return len(self._entity_type_to_field_names)

    @property
    def num_relations(self):
        return sum([len(x) for x in self._entity_type_to_relation_types.values()])

    def encode(self, datum):
        retval = {}
        for k, v in datum.items():
            if k not in self._field_name_to_object:
                retval[k] = v
            else:
                retval[k] = self._field_name_to_object[k].encode(v)
        return retval

    def decode(self, datum):
        retval = {}
        for k, v in datum.items():
            if k not in self._field_name_to_object:
                retval[k] = v
            else:
                try:
                    retval[k] = self._field_name_to_object[k].decode(v)
                except:
                    raise Exception("Could not decode {} value '{}' (spec={})".format(k, v, self._field_name_to_object[k]))
        return retval

    def decode_batch(self, batch):
        retvals = []
        #print(batch[None].shape)
        for i in range(len(batch[None])):
            cur = {}
            for field_name, values in batch.items():
                #print(field_name)
                try:
                    cur[field_name] = values[i].tolist()
                except:
                    pass
            #print(cur)
            dec = self.decode(cur)
            #print()
            retvals.append({k : v for k, v in dec.items() if k in list(self.entity_fields(dec[None])) + [None]})
        return retvals

    def encode_batch(self, batch):
        retval = {}
        for i in range(len(batch[None])):
            pass
        return retval


class Dataset(object):

    def __init__(self, spec, entities):
        self._spec = spec
        self._entities = []
        self._entity_fields = {}
        self._id_to_index = {}
        for idx, entity in enumerate(entities):
            entity_type = entity[self._spec.entity_type_field]
            entity_id = entity[self._spec.id_field]
            assert (entity_id not in self._id_to_index), "The id field ('{}') must be unique".format(self._spec.id_field)
            self._id_to_index[entity_id] = idx
            self._entity_fields[entity_type] = self._entity_fields.get(entity_type, set())
            for k in [x for x in entity.keys() if x != None]:
                self._entity_fields[entity_type].add(k)
            self._entities.append(entity)
        self._edges = {}
        for entity in self._entities:
            entity_type = entity[self._spec.entity_type_field]
            entity_id = entity[self._spec.id_field]
            source_index = self._id_to_index[entity_id]
            for relation_type in self._spec.entity_relations(entity_type):
                if relation_type in entity:
                    if entity[relation_type] not in self._id_to_index:
                        continue
                    target_index = self._id_to_index[entity[relation_type]]
                    self._edges[relation_type] = self._edges.get(relation_type, {})
                    self._edges[relation_type][source_index] = self._edges[relation_type].get(source_index, [])
                    self._edges[relation_type][source_index].append(target_index)
                    #self._edges[relation_type].append((source_index, target_index))
                #sys.exit()
        #for relation_type, relations in all_edges.items():
        #    self._edges[relation_type] = {}

            #rows, cols, vals = [], [], []
        #    for se, te in relations:
        #        self._edges[relation_type][se] = self._edges[relation_type].get(se, []) + [te]
                #rows.append(se)
                #cols.append(te)
                #vals.append(True)
            #self._edges[relation_type] = scipy.sparse.coo_matrix((vals, (rows, cols)), 
            #                                                     shape=(len(self), len(self)), dtype=numpy.bool)
        self._update_components()

    def subselect(self, indices):
        return Dataset(self._spec, [self._entities[i] for i in indices])

    def encode(self, item):
        return self._spec.encode(item)

    def decode(self, item):
        return self._spec.decode(item)

    def _update_components(self):
        rows, cols, vals = [], [], []
        for _, rs in self._edges.items():
            for r, cs in rs.items():
                for c in cs:
                    rows.append(r)
                    cols.append(c)
                    vals.append(True)
                    #adjacency = other if adjacency == None else adjacency.maximum(other)
        adjacency = scipy.sparse.csr_matrix((vals, (rows, cols)), 
                                            shape=(len(self), len(self)), 
                                            dtype=numpy.bool) #.todense()
        num, ids = connected_components(adjacency)
        components = {}    
        for i, c in enumerate(ids):
            components[c] = components.get(c, [])
            components[c].append(i)
        largest_component_size = 0 if len(components) == 0 else max([len(x) for x in components.values()])
        logging.info("Found %d connected components with maximum size %d", len(components), largest_component_size)
        #sys.exit()
        self._components = []
        for c in components.values():
            #component_adjacencies = {} #{k : numpy.full((len(c), len(c)), False) for k in self._edges.keys()}
            ca_rows, ca_cols = {}, {}
            g2l = {k : v for v, k in enumerate(c)}            
            for gsi in c:
                lsi = g2l[gsi]
                #print(gsi, lsi)
                for rel_type, rows in self._edges.items():
                    ca_rows[rel_type] = ca_rows.get(rel_type, [])
                    ca_cols[rel_type] = ca_cols.get(rel_type, [])
                    #print(rel_type)
                    for gti in rows.get(gsi, []):                        
                        lti = g2l[gti]
                        #print(9999, gti, lti)
                        ca_rows[rel_type].append(lsi)
                        ca_cols[rel_type].append(lti)
                        #component_adjacencies[rel_type][lsi, lti] = True
            component_adjacencies = {k : scipy.sparse.csr_matrix(([True for _ in ca_rows[rel_type]], (ca_rows[rel_type], ca_cols[rel_type])), 
                                                                 shape=(len(c), len(c)), dtype=numpy.bool) for k in self._edges.keys()}
            self._components.append((c, component_adjacencies)) # = [c for c in components.values()]
            #print(self._components[-1])
        #for c in range(self.num_components):
        #    ents, adjs = self.component(c)
        #    ne = len(ents)
        #    for k, v in adjs.items():
        #        a, b = v.shape
        #        assert(ne == a and ne == b)
        #    #print(len(ents), len(adjs))
        #sys.exit()
        #for c in self._components:
        #    print(c)
        #sys.exit()

        
    def __getitem__(self, index):
        return self._entities[index]
    
    def __len__(self):
        return len(self._entities)

    def component(self, i):
        entity_indices, adjacencies = self._components[i]
        entities = [self.encode(self[i]) for i in entity_indices]
        return (entities, adjacencies)
        #for k, v in adjacencies.items():
        #    assert(len(entities) == v.shape[0])
        #print(entities, adjacencies)
        sys.exit()
        return (entities, adjacencies)
        #retents = []
        #g2l = {k : v for v, k in enumerate(self._components[i])}
        #adj = {}
        for j in self._components[i]:
            retents.append(self.encode(self[j]))
            for rel_type, rels in self._edges.items():
                vals, rows, cols = [], [], []
                for k in rels.get(j, []):
                    vals.append(True)
                    rows.append(g2l[j])
                    cols.append(g2l[k])
                adj[rel_type] = scipy.sparse.csr_matrix((vals, (rows, cols)), 
                                                        shape=(len(retents), len(retents)), 
                                                        dtype=numpy.bool).todense()
        return (retents, adj)

    @property
    def entity_types(self):
        return self._entity_types
    
    @property
    def field_names(self):
        return [x for x in self._spec.keys() if isinstance(x, str)]

    @property
    def num_components(self):
        return len(self._components)

    def __str__(self):
        return "Dataset({} entities, {} components with max size {})".format(len(self),
                                                                             len(self._components),
                                                                             0 if len(self._components) == 0 else max([len(c) for c, _ in self._components]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--spec", dest="spec", help="Input file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.input, "rb") as ifd:
        data = pickle.load(ifd)

    with gzip.open(args.spec, "rb") as ifd:
        spec = pickle.load(ifd)

    import utils

    for i in range(data.num_components):
        batch_entities, batch_adjacencies = data.component(i)
        x = spec.field_object(None)["slave"]
        a = len([j for j in batch_entities if j[None] == x])
        b = (batch_adjacencies["slave_to_source"] == True).sum(1)
        assert(a == b.sum())
        assert(a == 1)
        #print(a, b.sum())


    comps = list(range(data.num_components))
    for batch_entities, batch_adjacencies in utils.batchify(data, comps, 128, subselect=False):
        #for i in range(data.num_components):
        #c, a = data.component(i)
        #print(spec.field_object(None)._rlookup) #["slave"]       
        x = spec.field_object(None)["slave"]
        a = batch_entities[None] == x
        b = (batch_adjacencies["slave_to_source"] == True).sum(1)
        #print(a.sum(), b.sum())
        #assert(a.tolist() == b.tolist())
    sys.exit()

    import utils

    #print(data)
    #print(data[0])
    #for entity in data.component(0)[0]:
    #    print(data.decode(entity))
    #print(data.component(0)[1])
    #sys.exit()
    for batch_entities, batch_adjacencies in utils.batchify(data, [0, 1, 2, 3], 16):
        #print(spec.decode_batch(batch_entities))
        continue
        for i in range(batch_entities[None].shape[0]):
            obj = {}
            for k, v in batch_entities.items():
                if v[i].sum() > 0:
                    obj[k] = v[i].tolist()
        #    print(data.decode(obj))
        #print(batch_adjacencies)
    #print(batch_entities)
