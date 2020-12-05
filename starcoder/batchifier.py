import logging
import numpy
import scipy.sparse
import torch
import functools
import argparse
from starcoder.random import random
from starcoder.configuration import Configurable
from starcoder.schema import Schema
from starcoder.dataset import Dataset
from starcoder.property import Property
from starcoder.entity import UnpackedEntity, PackedEntity, ID, Index, StackedEntities, stack_entities
from starcoder.adjacency import Adjacency, Adjacencies, stack_adjacencies
from abc import ABCMeta, abstractmethod
from typing import Type, List, Dict, Set, Any, Union, Tuple, Iterator
from torch import Tensor

logger = logging.getLogger(__name__)

def components_to_batch(comps: List[Tuple[List[UnpackedEntity], Adjacencies]], data: Dataset) -> Tuple[StackedEntities, Adjacencies]:
    entities: List[UnpackedEntity] = sum([es for es, _ in comps], [])    
    return (stack_entities([data.schema.pack(e) for e in entities], data.schema.properties),
            stack_adjacencies([a for _, a in comps]))

def dataset_to_batch(dataset: Dataset) -> Tuple[StackedEntities, Adjacencies]:
    return components_to_batch(
        [dataset.component(i) for i in range(dataset.num_components)],
        dataset
    )


class Batchifier(Configurable, metaclass=ABCMeta):
    def __init__(self, rest: Any) -> None:
        super(Batchifier, self).__init__(rest)
    @abstractmethod
    def __call__(self, data: "Dataset", batch_size: int) -> Iterator[Tuple[StackedEntities, Adjacencies]]: pass

    
class SampleEntities(Batchifier):
    arguments = [
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across batches"},
        {"dest" : "share_at_connectivity", "default" : 1.0, "type" : float, "help" : "Minimum percent-connectivity at which to always share an entity"},
    ]
    def __init__(self, vals: Any) -> None:
        super(SampleEntities, self).__init__(vals)
    def __call__(self, data: "Dataset", batch_size: int) -> Iterator[Tuple[StackedEntities, Adjacencies]]:
        entities_to_duplicate = data.get_type_ids(*self.shared_entity_types) # type: ignore
        other_entities = [i for i in data.ids if i not in entities_to_duplicate]
        num_other_entities = batch_size - len(entities_to_duplicate)
        assert num_other_entities > 0
        random.shuffle(other_entities)
        while len(other_entities) > 0:
            ids = entities_to_duplicate + other_entities[:num_other_entities]
            other_entities = other_entities[num_other_entities:]
            new_data = data.subselect_entities(ids)
            comps = [new_data.component(i) for i in range(new_data.num_components)]
            yield components_to_batch(comps, data)


class SampleComponents(Batchifier):
    arguments = [
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across batches"},
    ]
    def __init__(self, vals: Any) -> None:
        super(SampleComponents, self).__init__(vals)
    def __call__(self, data: "Dataset", batch_size: int) -> Iterator[Tuple[StackedEntities, Adjacencies]]:
        entities_to_duplicate = data.get_type_ids(*self.shared_entity_types) # type: ignore
        nonduplicate_entities_per_batch = batch_size - len(entities_to_duplicate)
        assert nonduplicate_entities_per_batch > 0
        if len(entities_to_duplicate) == 0:
            nonduplicate_data = data
        else:
            nonduplicate_data = data.subselect_entities([i for i in data.ids if i not in entities_to_duplicate])
        components = [i for i in range(nonduplicate_data.num_components)]
        random.shuffle(components)
        total = 0
        current_batch: List[ID] = []
        ids = []
        while len(components) > 0 or len(ids) > 0:
            if len(ids) == 0:
                ids = [i for i in nonduplicate_data.component_ids(components[0])]
                components = components[1:]
            if len(current_batch) + len(ids) > nonduplicate_entities_per_batch:
                if len(current_batch) > 0:
                    logger.debug("Returning batch of size %d", len(current_batch))
                    yield dataset_to_batch(data.subselect_entities(current_batch + entities_to_duplicate))
                total += len(current_batch)
                current_batch = ids[:nonduplicate_entities_per_batch]
                ids = ids[len(current_batch):]
            else:
                current_batch += ids
                ids = []
        if len(current_batch) > 0:
            logger.debug("Returning batch of size %d", len(current_batch))            
            yield dataset_to_batch(data.subselect_entities(current_batch + entities_to_duplicate))
            total += len(current_batch)
            

class SampleSnowflakes(Batchifier):
    arguments = [
        {"dest" : "seed_entity_type", "default" : None, "help" : "Entity type to sample from as initial seeds"},
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across batches"},
        {"dest" : "neighbor_sample_probability", "default" : 0.5, "help" : "Probability of sampling each neighbor"},
    ]    
    def __init__(self, rest: Any) -> None:
        super(SampleSnowflakes, self).__init__(rest)
    def __call__(self, data: "Dataset", batch_size: int) -> Iterator[Tuple[StackedEntities, Adjacencies]]:
        entities_to_duplicate = data.get_type_ids(*self.shared_entity_types) # type: ignore
        other_entity_ids = [i for i in data.ids if i not in entities_to_duplicate]
        num_other_entities = batch_size - len(entities_to_duplicate)
        assert num_other_entities > 0
        other_entities = data.subselect_entities(other_entity_ids)
        while len(other_entities) > 0:
            if len(other_entities) <= num_other_entities:
                batch = data.subselect_entities_by_id(list(other_entities.id_to_index.keys()) + entities_to_duplicate)
                other_entities = data.subselect_entities_by_id([])
            else:
                batch_entities = []
                other_components = [i for i in range(other_entities.num_components)]
                random.shuffle(other_components)
                while len(other_components) > 0:
                    comp_ids = other_entities.component_ids(other_components[0])
                    comp_adjs = [x.todense() for x in other_entities.component_adjacencies(other_components[0]).values()]
                    adjs = functools.reduce(lambda x, y : x | y, [numpy.full((len(comp_ids), len(comp_ids)), False)] + comp_adjs + [x.T for x in comp_adjs])
                    seed_num = adjs.sum(1).argmax()
                    other_components = other_components[1:]
                    hops = []
                    hops.append([seed_num])
                    for depth in range(3):
                        poss = numpy.argwhere(adjs[hops[-1]].any(0) == True)[:, 1].tolist()
                        random.shuffle(poss)
                        hops.append(poss[:len(poss) // 2])
                    nonshared_entities = [other_entities.index_to_id[i] for i in set(sum(hops, []))]
                    batch_entities += nonshared_entities
                batch_entities = list(set(batch_entities))[0:num_other_entities] + entities_to_duplicate

                batch = data.subselect_entities_by_id(batch_entities)
                other_entities = other_entities.subselect_entities_by_id(batch_entities, invert=True)
            comps = [batch.component(i) for i in range(batch.num_components)]
            yield components_to_batch(comps, data)
