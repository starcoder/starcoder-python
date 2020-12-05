from typing import Dict, List, Any, Tuple, Generator
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.schema import Schema
from starcoder.dataset import Dataset
from starcoder.entity import stack_entities, UnpackedEntity, PackedEntity, Index, ID
from starcoder.adjacency import Adjacencies
import torch
import numpy
import logging
import tempfile
import json
import gzip
import os
import random

logger = logging.getLogger("starcoder")

def simple_loss_policy(losses_by_property: Dict[str, List[float]]) -> float:
    loss_by_property = {k : sum(v) for k, v in losses_by_property.items()}
    retval = sum(loss_by_property.values())
    return retval

def compute_losses(model: GraphAutoencoder,
                   entities: Dict[str, Any],
                   reconstructions: Dict[str, Any],
                   schema: Schema) -> Dict[str, torch.Tensor]:
    """
    Gather and return all the property losses as a nested dictionary where the first
    level of keys are property types, and the second level are property names.
    """
    logger.debug("Started computing all losses")
    losses = {}
    for entity_type_name, entity_type in schema.entity_types.items():
        for property_name in entity_type.properties:
            property_values = entities[property_name]
            #for property_name, property_values in entities.items():
            #if property_name in schema.properties and property_name in reconstructions:
            property = schema.properties[property_name]
            logger.debug("Computing losses for property %s of type %s", property.name, property.type_name)
            reconstruction_values = reconstructions[property.name]
            property_losses = model.property_losses[property.name](reconstruction_values, property_values)
            #print(property_name, property_losses)
            mask = ~torch.isnan(property_losses)
            losses[property.name] = torch.masked_select(property_losses, mask)
    logger.debug("Finished computing all losses")
    return losses


def run_over_components(model: GraphAutoencoder,
                        batchifier: Any,
                        optim: Any,
                        loss_policy: Any,
                        data: Dataset,
                        batch_size: int,
                        gpu: bool,
                        train: bool,
                        property_dropout: float=0.0,
                        neuron_dropout: float=0.0,
                        subselect: bool=False,
                        strict: bool=True) -> Tuple[Any, Dict[Any, Any]]:
    old_mode = model.training
    model.train(train)
    loss_by_property: Dict[str, Any] = {}
    score_by_property: Dict[str, Any] = {}
    loss = 0.0
    for batch_num, (full_entities, full_adjacencies) in enumerate(batchifier(data, batch_size)):
        logger.debug("Processing batch #%d", batch_num)
        partial_entities = {}
        indices = list(range(full_entities[data.schema.entity_type_property.name].shape[0]))
        num_property_dropout = int(property_dropout * len(indices))

        for property_name, property_values in full_entities.items():
            random.shuffle(indices)
        
            if property_name in data.schema.properties and False:
                partial_entities[property_name] = property_values.clone()
                partial_entities[property_name][indices[:num_property_dropout]] = 0 if data.schema.properties[property_name].stacked_type == torch.int64 else float("nan")
                #data.schema.properties[property_name].missing_value
            else:
                partial_entities[property_name] = property_values

        batch_loss_by_property = {}
        if gpu:
            partial_entities = {k : v.cuda() if hasattr(v, "cuda") else v for k, v in partial_entities.items()}
            full_entities = {k : v.cuda() if hasattr(v, "cuda") else v for k, v in full_entities.items()}
            full_adjacencies = {k : v.cuda() for k, v in full_adjacencies.items()}
        optim.zero_grad()
        reconstructions, norm, bottlenecks = model(partial_entities, full_adjacencies)
        for property, losses in compute_losses(model, full_entities, reconstructions, data.schema).items():
            score_by_property[property] = score_by_property.get(property, [])
            golds = full_entities[property]
            #mask = (full_entities[field] != data.schema.data_fields[field].missing_value) & 
            #print(partial_entities[field])
            #print(field)
            #mask = (partial_entities[field] == data.schema.data_fields[field].missing_value) | partial_entities[field].isnan()
            #mask = mask & (full_entities[field] != data.schema.data_fields[field].missing_value)
            #print(mask.shape, mask.sum())

            # if data.schema.json["data_fields"][field]["type"] == "categorical":
            #     guesses = torch.argmax(reconstructions[field], 1)
            #     guesses = torch.masked_select(guesses, mask)
            #     golds = torch.masked_select(golds, mask)
            #     #equal = torch.masked_select(golds, mask) == torch.masked_select(guesses, mask)
            #     #print(equal.shape)
            #     #vals = [1 if x else 0 for x in equal.squeeze().tolist()]
            #     print(guesses.shape, golds.shape)
            #     vals = [1 if x else 0 for x in (guesses == golds).squeeze().tolist()]
            #     #val = 0.0 if equal.shape[0] == 0.0 else equal.sum() / float(equal.shape[0])
            # elif data.schema.json["data_fields"][field]["type"] == "scalar":
            #     guesses = reconstructions[field].squeeze()
            #     guesses = torch.masked_select(guesses, mask)
            #     golds = torch.masked_select(golds, mask)
            #     vals = ((golds - guesses)**2).squeeze().tolist()
            # score_by_field[field] += vals #[val] * mask.shape[0]
            if losses.shape[0] > 0:
                batch_loss_by_property[property] = losses            
        logging.debug("Applying loss policy")
        batch_loss = loss_policy(batch_loss_by_property)
        loss += batch_loss.clone().detach()
        if train:            
            logging.debug("Back-propagating")
            batch_loss.backward()
            logging.debug("Stepping")
            optim.step()
        logging.debug("Assembling losses by property")
        for property, v in batch_loss_by_property.items():
            loss_by_property[property] = loss_by_property.get(property, [])
            loss_by_property[property].append(v.clone().detach())
        logging.debug("Finished batch #%d", batch_num)
    model.train(old_mode)
    return (loss, loss_by_property, score_by_property)

def apply_to_components(model: GraphAutoencoder,
                        batchifier: Any,
                        data: Dataset,
                        batch_size: int,
                        gpu: bool,
                        mask_property,
                        mask_probability) -> Generator[Tuple[Any, Dict[Any, Any]], None, None]:
    masking = None
    old_mode = model.training
    model.train(False)
    for batch_num, (full_entities, full_adjacencies) in enumerate(batchifier(data, batch_size)):
        logger.info("Processing batch #%d", batch_num)
        #batch_loss_by_field = {}
        if gpu:
            full_entities = {k : v.cuda() if hasattr(v, "cuda") else v for k, v in full_entities.items()}
            full_adjacencies = {k : v.cuda() for k, v in full_adjacencies.items()}
        yield model(full_entities, full_adjacencies) + (masking,)
    model.train(old_mode)


# def apply_to_globally_masked_components(to_mask: List[str],
#                                         model: GraphAutoencoder,
#                                         batchifier: Any,
#                                         data: Dataset,
#                                         batch_size: int,
#                                         gpu: bool) -> Generator[Tuple[Any, Dict[Any, Any]], None, None]:
#     scores = [1.0]
#     old_mode = model.training
#     model.train(False)
#     for batch_num, (full_entities, full_adjacencies) in enumerate(batchifier(data, batch_size)):
#         logger.debug("Processing batch #%d", batch_num)        
#         #batch_loss_by_field = {}
#         masked_full_entities = {k : (v.copy() if isinstance(v, numpy.ndarray) else v.clone()) for k, v in full_entities.items()}
#         for field in to_mask:
#             print(field)
#             masked_full_entities[field] = torch.full_like(masked_full_entities[field], data.schema.data_fields[field].missing_value)
#         if gpu:
#             masked_full_entities = {k : v.cuda() if hasattr(v, "cuda") else v for k, v in masked_full_entities.items()}
#             full_adjacencies = {k : v.cuda() for k, v in full_adjacencies.items()}
#         ret = model(masked_full_entities, full_adjacencies)
#     model.train(old_mode)
#     return sum(scores) / len(scores)

# def apply_to_locally_masked_components(to_mask: List[str],
#                                        model: GraphAutoencoder,
#                                        batchifier: Any,
#                                        data: Dataset,
#                                        batch_size: int,
#                                        gpu: bool) -> Generator[Tuple[Any, Dict[Any, Any]], None, None]:
#     old_mode = model.training
#     model.train(False)
#     for batch_num, (full_entities, full_adjacencies) in enumerate(batchifier(data, batch_size)):
#         logger.debug("Processing batch #%d", batch_num)
#         #batch_loss_by_field = {}
#         if gpu:
#             full_entities = {k : v.cuda() if hasattr(v, "cuda") else v for k, v in full_entities.items()}
#             full_adjacencies = {k : v.cuda() for k, v in full_adjacencies.items()}
#         yield model(full_entities, full_adjacencies)
#     model.train(old_mode)


def run_epoch(model: GraphAutoencoder,
              batchifier: Any,
              optimizer: Any,
              loss_policy: Any,
              train_data: Dataset,
              dev_data: Dataset,
              batch_size: int,
              gpu: bool,
              train_property_dropout: float=0.0,
              train_neuron_dropout: float=0.0,
              dev_property_dropout: float=0.0,
              subselect: bool=False) -> Tuple[Any, Any, Any, Any]:
    model.train(True)
    logger.debug("Running over training data")
    train_loss, train_loss_by_property, train_score_by_property = run_over_components(
        model,
        batchifier,
        optimizer, 
        loss_policy,
        train_data, 
        batch_size, 
        gpu,
        subselect=subselect,
        property_dropout=train_property_dropout,
        neuron_dropout=train_neuron_dropout,
        train=True
    )
    logger.debug("Running over dev data")
    model.train(False)
    dev_loss, dev_loss_by_property, dev_score_by_property = run_over_components(
        model,
        batchifier,
        optimizer, 
        loss_policy,
        dev_data, 
        batch_size, 
        gpu,
        property_dropout=dev_property_dropout,
        subselect=subselect,
        train=False,
    )

    return (train_loss.clone().detach().cpu(),
            #train_loss_by_field,
            {k : sum([x.sum() for x in v]) for k, v in train_loss_by_property.items()},
            #{k : sum([x.sum() for x in v]) for k, v in dev_loss_by_field.items()},
            #{k : [v.clone().detach().cpu() for v in vv] for k, vv in train_loss_by_field.items()},
            dev_loss.clone().detach().cpu(),
            {k : sum([x.sum() for x in v]) for k, v in dev_loss_by_property.items()},
            train_score_by_property,
            dev_score_by_property,
            )
            #{k : [v.clone().detach().cpu() for v in vv] for k, vv in dev_loss_by_field.items()})

def apply_model(model: Any, data: Dataset, args: Any, schema: Schema, ofd: Any=None) -> Any:
    
    model.eval()
    model.train(False)
    
    num_batches = data.num_entities // args.batch_size
    num_batches = num_batches + 1 if num_batches * args.batch_size < data.num_entities else num_batches
    ids = data.ids
    batch_to_batch_ids = {b : [ids[i] for i in range(b * args.batch_size, (b + 1) * args.batch_size) if i < data.num_entities] for b in range(num_batches)}
    representation_storage = {}
    bottleneck_storage = {}
    logging.debug("Dataset has %d entities", data.num_entities)
    try:        
        for batch_num, batch_ids in batch_to_batch_ids.items():
            representation_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            bottleneck_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            packed_batch_entities: List[PackedEntity] = [schema.pack(data.entity(i)) for i in batch_ids]
            stacked_batch_entities = stack_entities(packed_batch_entities, data.schema.properties)
            encoded_batch_entities = model.encode_properties(stacked_batch_entities)
            entity_indices, property_indices, entity_property_indices = model.compute_indices(stacked_batch_entities)            
            encoded_entities = model.create_autoencoder_inputs(encoded_batch_entities, entity_indices)
            bottlenecks, outputs = model.run_first_autoencoder_layer(encoded_entities)
            torch.save((batch_ids, outputs, entity_indices), representation_storage[batch_num]) # type: ignore
            torch.save(bottlenecks, bottleneck_storage[batch_num]) # type: ignore
        for depth in range(1, model.depth + 1):
            bottlenecks = {}
            adjacencies: Adjacencies = {}
            bns = {}
            for batch_num, batch_ids in batch_to_batch_ids.items():
                for entity_type_name, bns in torch.load(bottleneck_storage[batch_num]).items(): # type: ignore
                    bottlenecks[entity_type_name] = bottlenecks.get(entity_type_name, torch.zeros(size=(data.num_entities, model.bottleneck_size)))
            for batch_num, batch_ids in batch_to_batch_ids.items():
                entity_ids, ae_inputs, entity_indices = torch.load(representation_storage[batch_num]) # type: ignore
                bottlenecks, outputs = model.run_structured_autoencoder_layer(depth, ae_inputs, bottlenecks, adjacencies, {}, entity_indices)
                
                torch.save((entity_ids, outputs, entity_indices), representation_storage[batch_num]) # type: ignore
                torch.save(bottlenecks, bottleneck_storage[batch_num]) # type: ignore

        for batch_num, b_ids in batch_to_batch_ids.items():
            logging.debug("Saving batch %d with %d entities", batch_num, len(b_ids))
            entity_ids, outputs, entity_indices = torch.load(representation_storage[batch_num]) # type: ignore
            bottlenecks = torch.load(bottleneck_storage[batch_num]) # type: ignore
            proj = torch.zeros(size=(len(b_ids), model.projected_size))                
            decoded_properties = model.decode_properties(model.project_autoencoder_outputs(outputs))
            decoded_entities = model.assemble_entities(decoded_properties, entity_indices)
            decoded_properties = {k : {kk : vv for kk, vv in v.items() if kk in data.schema.entity_types[k].properties} for k, v in decoded_properties.items()}
            ordered_bottlenecks = {}
            for entity_type, indices in entity_indices.items():
                for i, index in enumerate(indices):
                    ordered_bottlenecks[index.item()] = bottlenecks[entity_type][i]

            for i, eid in enumerate(b_ids):
                original_entity = data.entity(eid)                 
                entity_type_name = original_entity[data.schema.entity_type_property.name]
                entity_data_properties = data.schema.entity_types[entity_type_name].properties

                reconstructed_entity = {data.schema.entity_type_property.name : entity_type_name}
                for property_name in original_entity.keys():
                    if property_name in entity_data_properties:
                        reconstructed_entity[property_name] = decoded_entities[property_name][i].tolist()
                reconstructed_entity = data.schema.unpack(reconstructed_entity)
                entity = {"original" : original_entity,
                          # include missing-but-generated properties
                          "reconstruction" : {k : v for k, v in reconstructed_entity.items() if k in original_entity},
                          "bottleneck" : ordered_bottlenecks[i].tolist(),
                }
                if ofd:
                    ofd.write(json.dumps(entity) + "\n")
    except Exception as e:
        raise e
    finally:
        for tfname in list(bottleneck_storage.values()) + list(representation_storage.values()):
            try:
                os.remove(tfname)
            except Exception as e:
                logging.info("Could not clean up temporary file: %s", tfname)
                raise e


            
def apply_model_cached(model: Any, data: Dataset, args: Any, schema: Schema, ofd: Any=None) -> Any:
    
    model.eval()
    model.train(False)
    
    num_batches = data.num_entities // args.batch_size
    num_batches = num_batches + 1 if num_batches * args.batch_size < data.num_entities else num_batches
    ids = data.ids
    batch_to_batch_ids = {b : [ids[i] for i in range(b * args.batch_size, (b + 1) * args.batch_size) if i < data.num_entities] for b in range(num_batches)}
    representation_storage = {}
    bottleneck_storage = {}
    logging.debug("Dataset has %d entities", data.num_entities)
    try:        
        for batch_num, batch_ids in batch_to_batch_ids.items():
            representation_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            bottleneck_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            packed_batch_entities: List[PackedEntity] = [schema.pack(data.entity(i)) for i in batch_ids]
            stacked_batch_entities = stack_entities(packed_batch_entities, data.schema.properties)
            encoded_batch_entities = model.encode_properties(stacked_batch_entities)
            entity_indices, property_indices, entity_property_indices = model.compute_indices(stacked_batch_entities)            
            encoded_entities = model.create_autoencoder_inputs(encoded_batch_entities, entity_indices)
            bottlenecks, outputs = model.run_first_autoencoder_layer(encoded_entities)
            torch.save((batch_ids, outputs, entity_indices), representation_storage[batch_num]) # type: ignore
            torch.save(bottlenecks, bottleneck_storage[batch_num]) # type: ignore
        for depth in range(1, model.depth + 1):
            bottlenecks = {}
            adjacencies: Adjacencies = {}
            bns = {}
            for batch_num, batch_ids in batch_to_batch_ids.items():
                for entity_type_name, bns in torch.load(bottleneck_storage[batch_num]).items(): # type: ignore
                    bottlenecks[entity_type_name] = bottlenecks.get(entity_type_name, torch.zeros(size=(data.num_entities, model.bottleneck_size)))
            for batch_num, batch_ids in batch_to_batch_ids.items():
                entity_ids, ae_inputs, entity_indices = torch.load(representation_storage[batch_num]) # type: ignore
                bottlenecks, outputs = model.run_structured_autoencoder_layer(depth, ae_inputs, bottlenecks, adjacencies, {}, entity_indices)
                
                torch.save((entity_ids, outputs, entity_indices), representation_storage[batch_num]) # type: ignore
                torch.save(bottlenecks, bottleneck_storage[batch_num]) # type: ignore

        for batch_num, b_ids in batch_to_batch_ids.items():
            logging.debug("Saving batch %d with %d entities", batch_num, len(b_ids))
            entity_ids, outputs, entity_indices = torch.load(representation_storage[batch_num]) # type: ignore
            bottlenecks = torch.load(bottleneck_storage[batch_num]) # type: ignore
            proj = torch.zeros(size=(len(b_ids), model.projected_size))                
            decoded_properties = model.decode_properties(model.project_autoencoder_outputs(outputs))
            decoded_entities = model.assemble_entities(decoded_properties, entity_indices)
            decoded_properties = {k : {kk : vv for kk, vv in v.items() if kk in data.schema.entity_types[k].properties} for k, v in decoded_properties.items()}
            ordered_bottlenecks = {}
            for entity_type, indices in entity_indices.items():
                for i, index in enumerate(indices):
                    ordered_bottlenecks[index.item()] = bottlenecks[entity_type][i]

            for i, eid in enumerate(b_ids):
                original_entity = data.entity(eid)                 
                entity_type_name = original_entity[data.schema.entity_type_property.name]
                entity_data_properties = data.schema.entity_types[entity_type_name].properties

                reconstructed_entity = {data.schema.entity_type_property.name : entity_type_name}
                for property_name in original_entity.keys():
                    if property_name in entity_data_properties:
                        reconstructed_entity[property_name] = decoded_entities[property_name][i].tolist()
                reconstructed_entity = data.schema.unpack(reconstructed_entity)
                entity = {"original" : original_entity,
                          # include missing-but-generated properties
                          "reconstruction" : {k : v for k, v in reconstructed_entity.items() if k in original_entity},
                          "bottleneck" : ordered_bottlenecks[i].tolist(),
                }
                if ofd:
                    ofd.write(json.dumps(entity) + "\n")
    except Exception as e:
        raise e
    finally:
        for tfname in list(bottleneck_storage.values()) + list(representation_storage.values()):
            try:
                os.remove(tfname)
            except Exception as e:
                logging.info("Could not clean up temporary file: %s", tfname)
                raise e
