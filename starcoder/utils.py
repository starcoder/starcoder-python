import pickle
import re
import gzip
import sys
import argparse
import json
import random
import logging
import warnings
import numpy
import torch

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class Configurable(object):
    arguments = []
    
    def parse_known_args(self, rest):
        parser = argparse.ArgumentParser()
        for arg in self.arguments: #getattr(cls, "arguments", []):
            parser.add_argument("--{}".format(arg["dest"]), **arg)
        return parser.parse_known_args(rest)

    def get_parse_template(self):
        template = []
        for arg in getattr(self, "arguments", []) + [{"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : ""}]:
            template.append("--{0} ${{{1}}}".format(arg["dest"], arg["dest"].upper()))
        return " ".join(template)

    def __init__(self, rest):
        args, rest = self.parse_known_args(rest)
        for k, v in vars(args).items():
            assert re.match(r"^[a-zA-Z_]+$", k) != None
            setattr(self, k, v)


# def compute_losses(entities, reconstructions, schema):
#     """
#     Gather and return all the field losses as a nested dictionary where the first
#     level of keys are field types, and the second level are field names.
#     """
#     losses = {}
#     for field_name, field_values in entities.items():
#         if field_name in schema.data_fields:
#             logger.debug("Computing losses for field %s", field_name)
#             #mask = entity_values > 0
#             field_type = type(schema.data_fields[field_name])
#             reconstruction_values = reconstructions[field_name]
#             #print(field_name, entity_values.shape, reconstruction_values.shape)
#             #recon = torch.reshape(reconstruction_values, (reconstruction_values.shape[0], -1)).sum(1)
#             #selector = torch.masked_select(torch.arange(0, entity_values.shape[0]),
#             #                               ~torch.isnan(entity_values) & ~torch.isnan(recon))
#             losses[field_type] = losses.get(field_type, {})
#             #masked_entity_values = torch.index_select(entity_values, 0, selector) #~torch.isnan(entity_values))
#             #print(field_values, reconstruction_values)
#             #print(field_name, field_values.shape, reconstruction_values.shape)
#             #masked_reconstruction_values = torch.index_select(reconstruction_values, 0, selector) #~torch.isnan(entity_values))
#             #masked_field_losses =
#             field_losses = field_model_classes[field_type][2](reconstruction_values, field_values)
#             losses[field_type][field_name] = field_losses
#     return losses



# def run_over_components(model, batchifier, optim, loss_policy, data, batch_size, gpu, train, subselect=False, strict=True, mask_tests=[]):
#     old_mode = model.training
#     model.train(train)
#     loss_by_field = {}
#     loss = 0.0
#     for batch_num, ((full_entities, full_adjacencies), (masked_entities, masked_adjacencies)) in enumerate(batchifier(data, batch_size)):
#         logger.debug("Processing batch #%d", batch_num)
#         batch_loss_by_field = {}
#         if gpu:
#             entities = {k : v.cuda() for k, v in entities.items()}
#             adjacencies = {k : v.cuda() for k, v in adjacencies.items()}
#         optim.zero_grad()
#         reconstructions, bottlenecks, ae_pairs = model(masked_entities, masked_adjacencies)
#         for field_type, fields in compute_losses(full_entities, reconstructions, data.schema).items():
#             for field_name, losses in fields.items():
#                 batch_loss_by_field[(field_name, field_type)] = losses        
#         batch_loss = loss_policy(batch_loss_by_field, ae_pairs)
#         loss += batch_loss
#         if train:            
#             batch_loss.backward()
#             optim.step()
#         for k, v in batch_loss_by_field.items():
#             loss_by_field[k] = loss_by_field.get(k, [])
#             loss_by_field[k].append(v.clone().detach())
#     model.train(old_mode)
#     return (loss, loss_by_field)


# def run_epoch(model, batchifier, optimizer, loss_policy, train_data, dev_data, batch_size, gpu, mask_tests=[], subselect=False):
#     model.train(True)
#     logger.debug("Running over training data")
#     train_loss, train_loss_by_field = run_over_components(model,
#                                                           batchifier,
#                                                           optimizer, 
#                                                           loss_policy,
#                                                           train_data, 
#                                                           batch_size, 
#                                                           gpu,
#                                                           subselect=subselect,
#                                                           train=True,
#                                                           mask_tests=mask_tests,
#     )
#     logger.debug("Running over dev data")
#     model.train(False)
#     dev_loss, dev_loss_by_field = run_over_components(model,
#                                                       batchifier,
#                                                       optimizer, 
#                                                       loss_policy,
#                                                       dev_data, 
#                                                       batch_size, 
#                                                       gpu,
#                                                       subselect=subselect,
#                                                       train=False,
#                                                       mask_tests=mask_tests,
#     )

#     return (train_loss.clone().detach().cpu(), 
#             {k : [v.clone().detach().cpu() for v in vv] for k, vv in train_loss_by_field.items()},
#             dev_loss.clone().detach().cpu(),
#             {k : [v.clone().detach().cpu() for v in vv] for k, vv in dev_loss_by_field.items()})

def batch_to_list(batch):
    """
    Take a batch, which represents data in the R/pandas style of a dictionary where keys are field names
    and values are lists of length equal to batch size, and return a list of entities (sort of like
    running zip over all the values, except preserving the field names)
    """
    retval = []
    for i in range(list(batch.values())[0].shape[0]):
        item = {}
        for k, v in batch.items():
            item[k] = v[i].item()
        retval.append(item)
    return retval


def tensorize(vals, field_obj):
    if any([isinstance(v, list) for v in vals]):
        max_length = max([len(v) for v in vals if isinstance(v, list)])
        vals = [(v + ([0] * (max_length - len(v)))) if v != None else [0] * max_length for v in vals]
    elif any([isinstance(v, (str,)) for v in vals]):
        return numpy.array(vals)
    else:
        vals = [(field_obj.missing_value if v == None else v) for v in vals]    
    retval = torch.tensor(vals, dtype=field_obj.encoded_type)
    return retval


def split_batch(entities, adjacencies, count):
    """
Naively split a batch in two.
    """    
    ix = list(range(len(entities)))
    random.shuffle(ix)
    first_ix = ix[0:count]
    second_ix = ix[count:]
    first_entities = [entities[i] for i in first_ix]
    second_entities = [entities[i] for i in second_ix]
    first_adjacencies = {}
    second_adjacencies = {}
    for rel_type, adj in adjacencies.items():
        try:
            adjacencies[rel_type] = adj[first_ix, :][:, first_ix]
            adjacencies[rel_type] = adj[second_ix, :][:, second_ix]
        except Exception as e:
            raise e
    return ((first_entities, first_adjacencies), (second_entities, second_adjacencies))


def stack_batch(components, schema):
    lengths = [len(x) for x, _ in components]
    entities = sum([x for x, _ in components], [])
    adjacencies = [x for _, x in components]
    full_adjacencies = {}
    start = 0
    for l, adjs in zip(lengths, adjacencies):
        for name, adj in adjs.items():
            full_adjacencies[name] = full_adjacencies.get(name, numpy.full((len(entities), len(entities)), False))
            full_adjacencies[name][start:start + l, start:start + l] = adj.todense()
        start += l
    field_names = set(sum([[k for k in e.keys()] for e in entities], list(schema.data_fields.keys())))
    full_entities = {k : [] for k in field_names}
    for entity in entities:
        enc_entity = schema.encode(entity)
        for field_name in field_names:
            full_entities[field_name].append(enc_entity.get(field_name, None))
    full_entities = {k : numpy.array(v) if k not in schema.data_fields else tensorize(v, schema.data_fields[k]) for k, v in full_entities.items()}
    ne = len(entities)
    for k, v in full_adjacencies.items():
        a, b = v.shape
    return (full_entities, {k : torch.tensor(v) for k, v in full_adjacencies.items()})
