import torch
import logging
import tempfile
import json
import os
import importlib
import re
import numpy
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def starport(name):
    mod, cls = re.match(r"^(.*)\.([^\.]*)$", name).groups()
    property_encoder_class = getattr(importlib.import_module(mod), cls)
    return property_encoder_class


def stack_entities(encoded_entities, properties):
    """
    Converts a list of packed entities into pandas/R-like format ("stacked"), i.e. a dictionary of properties 
    where each property has a tensor/multi-dimensional array of numeric values equal in first dimension to 
    the number of entities, with float("nan") for missing items.    
    """
    property_names = set(sum([[k for k in e.keys()] for e in encoded_entities], []))
    full_entities: Dict[str, List[Any]] = {k : [] for k in property_names}
    for entity in encoded_entities:
        for property_name in property_names:
            if property_name in properties:                
                full_entities[property_name].append(
                    entity.get(property_name, properties[property_name].missing_value)
                )
            else:
                full_entities[property_name].append(entity.get(property_name, False))
    retval = {}
    for k, v in full_entities.items():
        retval[k] = numpy.array(v) if k not in properties else torch.tensor(v, dtype=properties[k].stacked_type)
    return retval


def stack_adjacencies(to_stack):
    retval = {}
    offset = 0
    for adjs in to_stack:
        step: int = 0
        for name, adj in adjs.items():
            step = adj.shape[0]
            retval[name] = retval.get(name, {"values" : [], "rows" : [], "columns" : []})
            for r, c in zip(*adj.nonzero()):
                retval[name]["values"].append(True)
                retval[name]["rows"].append(r + offset)
                retval[name]["columns"].append(c + offset)
        offset += step
    return {k : torch.tensor(csr_matrix((v["values"], (v["rows"], v["columns"])),
                                        shape=(offset, offset),
                                        dtype=bool).todense()) for k, v in retval.items()}


def simple_loss_policy(losses_by_property):
    loss_by_property = {k : sum(v) for k, v in losses_by_property.items()}
    retval = sum(loss_by_property.values())
    return retval


def run_over_components(model,
                        batchifier,
                        optim,
                        loss_policy,
                        data,
                        batch_size,
                        gpu,
                        train,
                        property_dropout,
                        subselect,
                        mask_properties=[],
                        strict=True):
    old_mode = model.training
    model.train(train)
    loss_by_property = {}
    score_by_property = {}
    total_loss = 0.0
    for batch_num, (full_entities, full_adjacencies) in enumerate(batchifier(data, batch_size)):
        logger.debug("Processing batch #%d", batch_num)
        partial_entities = [{k : v for k, v in e.items()} for e in full_entities]
        optim.zero_grad()
        originals, loss, reconstructions, bottlenecks = model(partial_entities, full_adjacencies)
        logging.debug("Applying loss policy")
        total_loss += loss.clone().detach()
        if train:            
            logging.debug("Back-propagating")
            loss.backward()
            logging.debug("Stepping")
            optim.step()
        logging.debug("Finished batch #%d", batch_num)
    model.train(old_mode)
    reconstructions = {}
    bottlenecks = {}
    return (total_loss, reconstructions, bottlenecks)


def apply_to_components(model,
                        batchifier,
                        data,
                        batch_size,
                        gpu,
                        mask_properties,
                        mask_probability):
    masking = None
    old_mode = model.training
    model.train(False)
    for batch_num, (full_entities, full_adjacencies) in enumerate(batchifier(data, batch_size)):
        logger.debug("Processing batch #%d", batch_num)
        yield model(full_entities, full_adjacencies)
    model.train(old_mode)


def run_epoch(model,
              batchifier,
              optimizer,
              loss_policy,
              train_data,
              dev_data,
              batch_size,
              gpu,
              train_property_dropout=0.0,
              train_neuron_dropout=0.0,
              dev_property_dropout=0.0,
              mask_properties=[],
              subselect=False):
    model.train(True)
    logger.info("Running over training data with batch size %d", batch_size)
    train_loss, _, _ = run_over_components(
        model,
        batchifier,
        optimizer, 
        loss_policy,
        train_data, 
        batch_size, 
        gpu,
        subselect=subselect,
        property_dropout=train_property_dropout,
        train=True,
        mask_properties=mask_properties,
    )
    logger.info("Running over dev data with batch size %d", batch_size)
    model.train(False)
    dev_loss, _, _ = run_over_components(
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
        mask_properties=mask_properties,
    )
    return (train_loss.clone().detach().cpu(),
            dev_loss.clone().detach().cpu(),
            )


def apply_model(model, data, args, schema, ofd=None):
    model.eval()
    model.train(False)    
    num_batches = data.num_entities // args.batch_size
    num_batches = num_batches + 1 if num_batches * args.batch_size < data.num_entities else num_batches
    ids = data.ids
    batch_to_batch_ids = {
        b : [
            ids[i] for i in range(b * args.batch_size, (b + 1) * args.batch_size) if i < data.num_entities
        ] for b in range(num_batches)
    }
    representation_storage = {}
    bottleneck_storage = {}
    logging.debug("Dataset has %d entities", data.num_entities)
    try:        
        for batch_num, batch_ids in batch_to_batch_ids.items():
            representation_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            bottleneck_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            packed_batch_entities = [schema.pack(data.entity(i)) for i in batch_ids]
            stacked_batch_entities = stack_entities(packed_batch_entities, data.schema.properties)
            encoded_batch_entities = model.encode_properties(stacked_batch_entities)
            entity_indices, property_indices, entity_property_indices = model.compute_indices(
                stacked_batch_entities
            )
            encoded_entities = model.create_autoencoder_inputs(encoded_batch_entities, entity_indices)
            bottlenecks, outputs = model.run_first_autoencoder_layer(encoded_entities)
            torch.save((batch_ids, outputs, entity_indices), representation_storage[batch_num])
            torch.save(bottlenecks, bottleneck_storage[batch_num])
        for depth in range(1, model.depth + 1):
            bottlenecks = {}
            adjacencies = {}
            bns = {}
            for batch_num, batch_ids in batch_to_batch_ids.items():
                for entity_type_name, bns in torch.load(bottleneck_storage[batch_num]).items():
                    bottlenecks[entity_type_name] = bottlenecks.get(
                        entity_type_name, 
                        torch.zeros(size=(data.num_entities, model.bottleneck_size))
                    )
            for batch_num, batch_ids in batch_to_batch_ids.items():
                entity_ids, ae_inputs, entity_indices = torch.load(
                    representation_storage[batch_num]
                )
                bottlenecks, outputs = model.run_structured_autoencoder_layer(
                    depth,
                    ae_inputs,
                    bottlenecks,
                    adjacencies,
                    {},
                    entity_indices
                )
                torch.save((entity_ids, outputs, entity_indices), representation_storage[batch_num])
                torch.save(bottlenecks, bottleneck_storage[batch_num])
        for batch_num, b_ids in batch_to_batch_ids.items():
            logging.debug("Saving batch %d with %d entities", batch_num, len(b_ids))
            entity_ids, outputs, entity_indices = torch.load(representation_storage[batch_num])
            bottlenecks = torch.load(bottleneck_storage[batch_num])
            proj = torch.zeros(size=(len(b_ids), model.projected_size))                
            decoded_properties = model.decode_properties(model.project_autoencoder_outputs(outputs))
            decoded_entities = model.assemble_entities(decoded_properties, entity_indices)
            decoded_properties = {
                k : {
                    kk : vv for kk, vv in v.items() if kk in data.schema.entity_types[k].properties
                } for k, v in decoded_properties.items()
            }
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

            
def apply_model_cached(model, data, args, schema, ofd=None):
    model.eval()
    model.train(False)    
    num_batches = data.num_entities // args.batch_size
    num_batches = num_batches + 1 if num_batches * args.batch_size < data.num_entities else num_batches
    ids = data.ids
    batch_to_batch_ids = {
        b : [
            ids[i] for i in range(b * args.batch_size, (b + 1) * args.batch_size) if i < data.num_entities
        ] for b in range(num_batches)
    }
    representation_storage = {}
    bottleneck_storage = {}
    logging.debug("Dataset has %d entities", data.num_entities)
    try:        
        for batch_num, batch_ids in batch_to_batch_ids.items():
            representation_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            bottleneck_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            packed_batch_entities = [schema.pack(data.entity(i)) for i in batch_ids]
            stacked_batch_entities = stack_entities(packed_batch_entities, data.schema.properties)
            encoded_batch_entities = model.encode_properties(stacked_batch_entities)
            entity_indices, property_indices, entity_property_indices = model.compute_indices(
                stacked_batch_entities
            )
            encoded_entities = model.create_autoencoder_inputs(encoded_batch_entities, entity_indices)
            bottlenecks, outputs = model.run_first_autoencoder_layer(encoded_entities)
            torch.save((batch_ids, outputs, entity_indices), representation_storage[batch_num])
            torch.save(bottlenecks, bottleneck_storage[batch_num])
        for depth in range(1, model.depth + 1):
            bottlenecks = {}
            adjacencies = {}
            bns = {}
            for batch_num, batch_ids in batch_to_batch_ids.items():
                for entity_type_name, bns in torch.load(bottleneck_storage[batch_num]).items():
                    bottlenecks[entity_type_name] = bottlenecks.get(
                        entity_type_name, 
                        torch.zeros(size=(data.num_entities, model.bottleneck_size))
                    )
            for batch_num, batch_ids in batch_to_batch_ids.items():
                entity_ids, ae_inputs, entity_indices = torch.load(representation_storage[batch_num])
                bottlenecks, outputs = model.run_structured_autoencoder_layer(
                    depth,
                    ae_inputs,
                    bottlenecks,
                    adjacencies,
                    {},
                    entity_indices
                )                
                torch.save((entity_ids, outputs, entity_indices), representation_storage[batch_num])
                torch.save(bottlenecks, bottleneck_storage[batch_num])
        for batch_num, b_ids in batch_to_batch_ids.items():
            logging.debug("Saving batch %d with %d entities", batch_num, len(b_ids))
            entity_ids, outputs, entity_indices = torch.load(representation_storage[batch_num])
            bottlenecks = torch.load(bottleneck_storage[batch_num])
            proj = torch.zeros(size=(len(b_ids), model.projected_size))                
            decoded_properties = model.decode_properties(model.project_autoencoder_outputs(outputs))
            decoded_entities = model.assemble_entities(decoded_properties, entity_indices)
            decoded_properties = {
                k : {
                    kk : vv for kk, vv in v.items() if kk in data.schema.entity_types[k].properties
                } for k, v in decoded_properties.items()
            }
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
