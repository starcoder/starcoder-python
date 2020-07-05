import random
import argparse
import logging
from starcoder.utils import Configurable


logger = logging.getLogger(__name__)


class Splitter(Configurable):
    def __init__(self, rest):
        super(Splitter, self).__init__(rest)        
        total = sum(self.proportions)
        self.proportions = [p / total for p in self.proportions]
    def __call__(self, data):
        raise UnimplementedException()


class SampleEntities(Splitter):
    arguments = [
        {"dest" : "proportions", "nargs" : "*", "default" : [], "type" : float, "help" : "List of data proportions"},
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across splits"},
    ]    
    def __init__(self, rest):
        super(SampleEntities, self).__init__(rest)
    def __call__(self, data):
        to_duplicate = data.get_type_indices(*self.shared_entity_types)
        logger.info("Always including %d entities", len(to_duplicate))
        indices = [i for i in range(len(data)) if i not in to_duplicate]
        random.shuffle(indices)
        num_indices = len(indices)
        for num in [int(p * num_indices) for p in self.proportions]:
            yield indices[:num] + to_duplicate
            indices = indices[num:]


class SampleComponents(Splitter):
    arguments = [
        {"dest" : "proportions", "nargs" : "*", "default" : [], "type" : float, "help" : "List of data proportions"},
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : ""},
    ]    
    def __init__(self, rest):
        super(SampleComponents, self).__init__(rest)
    def __call__(self, data):
        shared_entities = data.get_type_indices(*self.shared_entity_types)
        logger.info("Always including %d entities", len(shared_entities))
        without_shared = data.subselect_entities_by_index([i for i in range(len(data)) if i not in shared_entities])
        component_indices = [i for i in range(without_shared.num_components)]
        logger.info("Without shared entities there are %d components", len(component_indices))
        random.shuffle(component_indices)
        num_indices = len(component_indices)
        for num in [int(p * num_indices) for p in self.proportions]:
            other_indices = sum([without_shared.component_indices(ci) for ci in component_indices[:num]], [])
            component_indices = component_indices[num:]
            yield [data.id_to_index[without_shared.index_to_id[i]] for i in other_indices] + shared_entities


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()
