import logging
import random
import numpy
import scipy.sparse
import torch
import functools
import argparse
from starcoder.utils import Configurable, tensorize, split_batch, stack_batch


logger = logging.getLogger(__name__)


class Batchifier(Configurable):
    def __init__(self, rest):
        super(Batchifier, self).__init__(rest)
    def __call__(self, data, batch_size):
        raise UnimplementedException()    

    
class SampleEntities(Batchifier):
    arguments = [{"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across batches"}]
    def __init__(self, vals):
        super(SampleEntities, self).__init__(vals)
    def __call__(self, data, batch_size):
        entities_to_duplicate = data.get_type_indices(*self.shared_entity_types)
        other_entities = [i for i in range(len(data)) if i not in entities_to_duplicate]
        num_other_entities = batch_size - len(entities_to_duplicate)
        assert num_other_entities > 0
        random.shuffle(other_entities)
        while len(other_entities) > 0:
            indices = entities_to_duplicate + other_entities[:num_other_entities]
            other_entities = other_entities[num_other_entities:]
            new_data = data.subselect_entities_by_index(indices)
            comps = [new_data.component(i) for i in range(new_data.num_components)]
            retval = stack_batch(comps, data.schema)
            logger.debug("Returning batch of size %d", len(new_data))
            yield retval


class SampleComponents(Batchifier):
    arguments = [{"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across batches"}]
    def __init__(self, vals):
        super(SampleComponents, self).__init__(vals)
    def __call__(self, data, batch_size):
        entities_to_duplicate = data.get_type_indices(*self.shared_entity_types)
        num_other_entities_per_batch = batch_size - len(entities_to_duplicate)
        assert num_other_entities_per_batch > 0
        other_entities = data.subselect_entities_by_index([i for i in range(len(data)) if i not in entities_to_duplicate])
        other_components = [i for i in range(other_entities.num_components)]
        random.shuffle(other_components)

        this_batch = []        
        while len(other_components) > 0:
            while len(this_batch) < num_other_entities_per_batch and len(other_components) > 0:
                indices = [data.id_to_index[other_entities.index_to_id[i]] for i in other_entities.component_indices(other_components[0])]
                other_components = other_components[1:]
                if len(this_batch) + len(indices) < num_other_entities_per_batch:
                    this_batch += indices
                if len(this_batch) + len(indices) > num_other_entities_per_batch:
                    new_data = data.subselect_entities_by_index(this_batch + indices[:num_other_entities_per_batch - len(this_batch)])
                    this_batch = indices[num_other_entities_per_batch - len(this_batch):]
                    comps = [new_data.component(i) for i in range(new_data.num_components)]
                    retval = stack_batch(comps, data.schema)
                    #logger.info("Returning batch of size %d", len(new_data))
                    yield retval
        if len(this_batch) > 0:
            new_data = data.subselect_entities_by_index(this_batch + indices[:num_other_entities_per_batch - len(this_batch)])
            this_batch = indices[num_other_entities_per_batch - len(this_batch):]
            comps = [new_data.component(i) for i in range(new_data.num_components)]
            retval = stack_batch(comps, data.schema)
            #logger.info("Returning batch of size %d", len(new_data))            
            yield retval
            

class SampleSnowflakes(Batchifier):
    arguments = [
        {"dest" : "seed_entity_type", "default" : None, "help" : "Entity type to sample from as initial seeds"},
        {"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : "Entity types to be shared across batches"},
        {"dest" : "neighbor_sample_probability", "default" : 0.5, "help" : "Probability of sampling each neighbor"},
    ]    
    def __init__(self, rest):
        super(SampleSnowflakes, self).__init__(rest)
    def __call__(self, data, batch_size):
        entities_to_duplicate = [data.index_to_id[i] for i in data.get_type_indices(*self.shared_entity_types)]
        other_entities = [i for i in data.id_to_index.keys() if i not in entities_to_duplicate]
        num_other_entities = batch_size - len(entities_to_duplicate)
        assert num_other_entities > 0
        other_entities = data.subselect_entities_by_id(other_entities) #[i for i in range(len(data)) if i not in entities_to_duplicate])
        while len(other_entities) > 0:
            if len(other_entities) <= num_other_entities:
                batch = data.subselect_entities_by_id(list(other_entities.id_to_index.keys()) + entities_to_duplicate)
                other_entities = []
            else:
                batch_entities = [] #[i for i in entities_to_duplicate]
                other_components = [i for i in range(other_entities.num_components)]
                random.shuffle(other_components)
                while len(other_components) > 0:
                    comp_ids = [data.index_to_id[i] for i in other_entities.component_indices(other_components[0])]
                    comp_adjs = [x.todense() for x in other_entities.component_adjacencies(other_components[0]).values()]
                    adjs = functools.reduce(lambda x, y : x | y, [numpy.full((len(comp_ids), len(comp_ids)), False)] + comp_adjs + [x.T for x in comp_adjs])
                    seed_num = adjs.sum(1).argmax()
                    other_components = other_components[1:]
                    hops = []
                    hops.append([seed_num])
                    for depth in range(3):
                        poss = numpy.argwhere(adjs[hops[-1]].any(0) == True)[:, 1].tolist()
                        random.shuffle(poss)
                        #print(len(poss))
                        hops.append(poss[:len(poss) // 2])
                    nonshared_entities = [other_entities.index_to_id[i] for i in set(sum(hops, []))]
                    #print(len(nonshared_entities))
                    batch_entities += nonshared_entities #[other_entities.index_to_id[i] for i in set(sum(hops, []))]
                batch_entities = list(set(batch_entities))[0:num_other_entities] + entities_to_duplicate

                batch = data.subselect_entities_by_id(batch_entities)
                other_entities = other_entities.subselect_entities_by_id(batch_entities, invert=True)
                #print(len(batch_entities))
                #print(len(other_entities))
                
            comps = [batch.component(i) for i in range(batch.num_components)]
            retval = stack_batch(comps, data.schema)                
            yield retval
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()
