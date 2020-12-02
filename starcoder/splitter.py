from starcoder.random import random
import argparse
import logging
#from starcoder.base import Splitter

from abc import ABCMeta, abstractmethod
from typing import Type, List, Dict, Set, Any, Callable, Iterator
from starcoder.base import StarcoderObject
from starcoder.entity import ID, Index
from starcoder.dataset import Dataset

logger = logging.getLogger(__name__)

class Splitter(StarcoderObject, metaclass=ABCMeta):
    def __init__(self, rest: Any) -> None:
        super(Splitter, self).__init__(rest)
        #total = sum(self.proportions)
        #self.proportions = [p / total for p in self.proportions]
    @abstractmethod
    def __call__(self, data: Any) -> Iterator[Any]: pass

class SampleEntities(Splitter):
    arguments = [
        {"dest" : "proportions", "nargs" : "*", "default" : [], "type" : float, "help" : "List of data proportions"},
        {"dest" : "share_at_connectivity", "default" : 1.0, "type" : float, "help" : "Minimum percent-connectivity at which to always share an entity"},
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across splits"},
    ]    
    def __init__(self, rest: Any) -> None:
        super(SampleEntities, self).__init__(rest)
    def __call__(self, data: Any) -> Iterator[Any]:
        to_duplicate = data.get_type_indices(*self.shared_entity_types) # type: ignore
        logger.info("Always including %d entities", len(to_duplicate))
        indices = [i for i in range(len(data)) if i not in to_duplicate]
        random.shuffle(indices)
        num_indices = len(indices)
        for num in [int(p * num_indices) for p in self.proportions]: # type: ignore
            yield indices[:num] + to_duplicate
            indices = indices[num:]


class SampleComponents(Splitter):
    arguments = [
        {"dest" : "proportions", "nargs" : "*", "default" : [], "type" : float, "help" : "List of data proportions"},
        {"dest" : "shared_entity_threshold", "default" : None, "type" : float, "help" : ""},        
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : ""},
    ]    
    def __init__(self, rest: Any) -> None:
        super(SampleComponents, self).__init__(rest)
    def __call__(self, data: Dataset) -> Iterator[Any]:
        #thresh = 15
        #adj = data.aggregate_adjacencies()
        #shared_entities = [i for i, j in enumerate(adj.sum(0).tolist()[0]) if j > thresh] + [i for i, j in enumerate(adj.sum(1).tolist()[0]) if j > thresh]
        shared_entities = data.get_type_ids(*self.shared_entity_types) # type: ignore
        logger.info("Always including %d entities", len(shared_entities))
        without_shared = data.subselect_entities_by_id([i for i in data.ids if i not in shared_entities])
        component_indices = [i for i in range(without_shared.num_components)]
        logger.info("Without shared entities there are %d components", len(component_indices))
        random.shuffle(component_indices)
        num_indices = len(component_indices)
        for num in [int(p * num_indices) for p in self.proportions]: # type: ignore
            logging.info("Selecting %d components for a split", num)
            other_ids: List[ID] = sum([without_shared.component_ids(ci) for ci in component_indices[:num]], [])
            logging.info("Reducing available components")
            component_indices = component_indices[num:]
            logging.info("Yielding split")
            yield other_ids + shared_entities #[data.id_to_index[without_shared.index_to_id[i]] for i in other_indices] + shared_entities
