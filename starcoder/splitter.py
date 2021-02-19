from starcoder.random import random
import logging

from abc import ABCMeta, abstractmethod
from typing import Type, List, Dict, Set, Any, Callable, Iterator
from starcoder.base import StarcoderObject
#from starcoder.entity import ID, Index
from starcoder.dataset import Dataset
from starcoder.utils import starport

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
    def __call__(self, schema, entities) -> Iterator[Any]:        
        id_property = schema["id_property"]
        entity_type_property = schema["entity_type_property"]
        shared_entity_types = schema["meta"]["shared_entity_types"]
        shared_entities = [e[id_property] for e in entities if e[entity_type_property] in shared_entity_types]
        logger.info("Always including %d shared entities", len(shared_entities))
        non_shared = [e[id_property] for e in entities if e[id_property] not in shared_entities]
        random.shuffle(non_shared)
        count = len(non_shared)
        proportions = schema["meta"]["proportions"]
        total = sum(proportions)
        proportions = [p / total for p in proportions]
        for num in [int(p * count) for p in proportions[:-1]]: # type: ignore
            yield non_shared[:num] + shared_entities
            non_shared = non_shared[num:]
        yield non_shared + shared_entities

class SampleEntities(Splitter):
    arguments = [
        {"dest" : "proportions", "nargs" : "*", "default" : [], "type" : float, "help" : "List of data proportions"},
        {"dest" : "share_at_connectivity", "default" : 1.0, "type" : float, "help" : "Minimum percent-connectivity at which to always share an entity"},
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across splits"},
    ]    
    def __init__(self, rest: Any) -> None:
        super(SampleEntities, self).__init__(rest)
    def __call__(self, schema, data: Any) -> Iterator[Any]:
        to_duplicate = data.get_type_ids(*self.shared_entity_types) # type: ignore
        logger.info("Always including %d entities", len(to_duplicate))
        indices = [i for i in range(len(data)) if i not in to_duplicate]
        random.shuffle(indices)
        num_indices = len(indices)
        for num in [int(p * num_indices) for p in self.proportions]: # type: ignore
            yield indices[:num] + to_duplicate
            indices = indices[num:]


class _SampleComponents(Splitter):
    arguments = [
        {"dest" : "proportions", "nargs" : "*", "default" : [], "type" : float, "help" : "List of data proportions"},
        {"dest" : "shared_entity_threshold", "default" : None, "type" : float, "help" : ""},        
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : ""},
    ]    
    def __init__(self, rest: Any) -> None:
        super(SampleComponents, self).__init__(rest)
    def __call__(self, schema, entities) -> Iterator[Any]:
        #thresh = 15
        #adj = data.aggregate_adjacencies()
        #shared_entities = [i for i, j in enumerate(adj.sum(0).tolist()[0]) if j > thresh] + [i for i, j in enumerate(adj.sum(1).tolist()[0]) if j > thresh]
        shared_entities = [e["id_property"] for e in entities if e[schema["entity_type_property"]] in self.shared_entity_types]
        #shared_entities = data.get_type_ids(*self.shared_entity_types) # type: ignore
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


class SampleComponents(Splitter):
    arguments = [
        {"dest" : "proportions", "nargs" : "*", "default" : [], "type" : float, "help" : "List of data proportions"},
        {"dest" : "shared_entity_threshold", "default" : None, "type" : float, "help" : ""},        
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : ""},
    ]    
    def __init__(self, rest: Any) -> None:
        super(SampleComponents, self).__init__(rest)
    def __call__(self, schema, data: Dataset) -> Iterator[Any]:
        shared_entity_types = schema["meta"]["shared_entity_types"]
        shared_entities = data.get_type_ids(*shared_entity_types)
        logger.info("Always including %d entities", len(shared_entities))
        without_shared = data.subselect_entities_by_id([i for i in data.ids if i not in shared_entities], strict=False)
        component_indices = [i for i in range(without_shared.num_components)]
        logger.info("Without shared entities there are %d components", len(component_indices))
        random.shuffle(component_indices)
        num_indices = len(component_indices)
        proportions = schema["meta"]["proportions"]
        proportions = [p / sum(proportions) for p in proportions]
        for num in [int(p * num_indices) for p in proportions[:-1]]: # type: ignore
            logging.info("Selecting %d components for a split", num)
            other_ids: List[ID] = sum([without_shared.component_ids(ci) for ci in component_indices[:num]], [])
            logging.info("Reducing available components")
            component_indices = component_indices[num:]
            logging.info("Yielding split")
            yield other_ids + shared_entities 
        logging.info("Selecting %d components for a split", len(component_indices))
        other_ids: List[ID] = sum([without_shared.component_ids(ci) for ci in component_indices], [])
        logging.info("Yielding split")
        yield other_ids + shared_entities 

if __name__ == "__main__":

    import argparse
    import gzip
    import json
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Full data file of one JSON object (entity) per line")
    parser.add_argument("--schema", dest="schema", help="Schema file")
    parser.add_argument("--random_seed", dest="random_seed", type=int, help="Random seed")
    parser.add_argument(dest="outputs", nargs="*", default=[], 
                        help="Sequence of output filenames corresponding to split proportions")
    parser.add_argument("--log_level",
                        dest="log_level",
                        default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
                        help="Logging level")
    args, rest = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )

    if args.random_seed != None:
        random.seed(args.random_seed)

    entities = []
    with (gzip.open if args.input.endswith("gz") else open)(args.input, "rt") as ifd:
        for line in ifd:
            entities.append(json.loads(line))

    with (gzip.open if args.schema.endswith("gz") else open)(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())

    data = Dataset(schema, entities)
    logging.info(
        "Loaded dataset with %d entities and %d components",
        len(data),
        data.num_components
    )
    
    sampler = starport(schema["meta"]["splitter"])(schema)
    for fname, split in zip(args.outputs, sampler(schema, data)):
        with (gzip.open if fname.endswith("gz") else open)(fname, "wt") as ofd:
            ofd.write(json.dumps(split, indent=2))
