import logging
import scipy.sparse
from scipy.sparse.csgraph import connected_components
import torch

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """
    The Dataset class is needed mainly for operations that depend on
    graph structure, particularly those that require connected components.
    A Dataset is, basically, a Schema and a list of JSON objects.
    """
    def __init__(self, schema, entities, strict=True):
        super(Dataset, self).__init__()
        self.schema = schema
        known_property_names = sum(
            [
                list(schema["properties"].keys()),
                list(schema["relationships"].keys()),
                [schema["id_property"], schema["entity_type_property"]]
            ],
            []
        )
        self.entities = []
        self.id_to_index = {}
        self.index_to_id = {}
        for idx, entity in [(x, y) for x, y in enumerate(entities)]:
            entity_id = entity[self.schema["id_property"]]
            if entity_id in self.id_to_index:
                raise Exception("Entity with id '{}' already exists".format(entity_id))
            self.id_to_index[entity_id] = idx
            self.index_to_id[idx] = entity_id
            for k in entity.keys():
                if k not in known_property_names:
                    raise Exception("Unknown property: '{}'".format(k))
            self.entities.append(entity)
        self.edges = {r : {} for r in self.schema["relationships"].keys()}
        for entity in self.entities:
            entity_type = entity[self.schema["entity_type_property"]]
            if entity_type not in self.schema["entity_types"]:
                raise Exception("Unknown entity type: '{}'".format(entity_type))
            entity_id = entity[self.schema["id_property"]]
            source_index = self.id_to_index[entity_id]            
            for relationship in [r for r, s in 
                                 self.schema["relationships"].items() 
                                 if s["source_entity_type"] == entity_type]:
                target_ids = entity.get(relationship, [])
                for target in target_ids if isinstance(target_ids, list) else [target_ids]:
                    if target not in self.id_to_index:
                        if strict:
                            raise Exception(
                                "Could not find target '{}' for entity '{}' relationship '{}'".format(
                                    target,
                                    entity_id,
                                    relationship
                                )
                            )
                        continue
                    target_index = self.id_to_index[target]
                    
                    self.edges[relationship][source_index] = self.edges[relationship].get(source_index, [])
                    self.edges[relationship][source_index].append(target_index)
        self.update_components()
        
    def get_type_ids(self, *type_names):
        retval = []
        for i in self.ids:
            if self.entity(i)[self.schema["entity_type_property"]] in type_names:
                retval.append(i)
        return retval

    def subselect_entities_by_id(self, ids, invert=False, strict=True):
        if invert:
            data = Dataset(
                self.schema, 
                [self.entities[i] for i in range(len(self)) if self.index_to_id[i] not in ids], 
                strict=True
            )
        else:
            # this should never fail lookup, but...also, commit or don't to always using strings
            data = Dataset(
                self.schema, 
                [self.entities[self.id_to_index[eid]] for eid in ids if eid in self.id_to_index], 
                strict=strict
            )
        return data
    
    def subselect_entities(self, ids, invert=False):
        return self.subselect_entities_by_id(ids, invert)

    def subselect_components(self, indices):
        return Dataset(
            self.schema, 
            [self.entities[self.id_to_index[eid]] for eid in sum([self.components[i][0] for i in indices], [])]
        )

    @property
    def ids(self):
        return [i for i in self.id_to_index.keys()]

    @property
    def aggregate_adjacencies(self):
        rows, cols, vals = [], [], []
        for _, rs in self.edges.items():
            for r, cs in rs.items():
                for c in cs:
                    rows.append(r)
                    cols.append(c)
                    vals.append(True)
        adjacency = scipy.sparse.csr_matrix((vals, (rows, cols)), 
                                            shape=(self.num_entities, self.num_entities), 
                                            dtype=bool)
        return adjacency
        
    def update_components(self):
        adjacency = self.aggregate_adjacencies
        num, ids = connected_components(adjacency)
        components_to_indices = {}    
        for i, c in enumerate(ids):
            components_to_indices[c] = components_to_indices.get(c, [])
            components_to_indices[c].append(i)
        if len(components_to_indices) == 0:
            raise Exception("This dataset is empty!")        
        self.components = []
        for indices in components_to_indices.values():
            ca_rows = {}
            ca_cols = {}
            g2l = {k : v for v, k in enumerate(indices)}
            for gsi in indices:
                lsi = g2l[gsi]
                for rel_type, rows in self.edges.items():
                    ca_rows[rel_type] = ca_rows.get(rel_type, [])
                    ca_cols[rel_type] = ca_cols.get(rel_type, [])
                    for gti in rows.get(gsi, []):
                        lti = g2l[gti]
                        ca_rows[rel_type].append(lsi)
                        ca_cols[rel_type].append(lti)
            component_adjacencies = {
                rel_type : scipy.sparse.csr_matrix(
                    (
                        [True for _ in ca_rows[rel_type]],
                        (ca_rows[rel_type], ca_cols[rel_type])
                    ),
                    shape=(len(indices), len(indices)),
                    dtype=bool
                ) for rel_type in self.edges.keys()
            }
            self.components.append(
                (
                    [self.index_to_id[c] for c in indices],
                    component_adjacencies
                )
            )

    @property
    def num_components(self):
        return len(self.components)

    @property
    def num_entities(self):
        return len(self.entities)
    
    def component_ids(self, i):
        return self.components[i][0]

    def component_adjacencies(self, i):
        return self.components[i][1]

    def entity(self, i):
        return self.entities[self.id_to_index[i]]
    
    def component(self, i):
        entity_ids, adjacencies = self.components[i]
        return (
            [self.entity(i) for i in entity_ids],
            adjacencies.copy()
        )

    def __getitem__(self, i):
        return self.entities[i]

    def __len__(self):
        return len(self.entities)

    def __str__(self) -> str:
        return "Dataset({} entities, {} components)".format(self.num_entities,
                                                            self.num_components)


if __name__ == "__main__":
    pass
