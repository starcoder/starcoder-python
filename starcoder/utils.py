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
        logger.info("Processing batch #%d (%s) (%s)", batch_num, len(full_entities), data.num_entities)
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

            
def apply_model_with_cache(model, data, batch_size, use_gpu, ofd=None):
    """
    Ugly code to run an exact forward pass on large components by caching
    intermediary representations and bottlenecks.
    """
    model.eval()
    model.train(False)    
    num_batches = data.num_entities // batch_size
    num_batches = num_batches + 1 if num_batches * batch_size < data.num_entities else num_batches
    ids = data.ids
    batch_to_batch_ids = {
        b : [
            (i, ids[i]) for i in range(b * batch_size, (b + 1) * batch_size) if i < data.num_entities
        ] for b in range(num_batches)
    }
    representation_storage = {}
    bottleneck_storage = {}

    adjacencies = {}
    entity_indices = {k : data.get_type_indices(k) for k in data.schema["entity_types"].keys()}
    edges = data.edges
    for rel_type, rel_spec in data.schema["relationships"].items():
        source = rel_spec["source_entity_type"]
        target = rel_spec["target_entity_type"]
        adjacencies[rel_type] = torch.zeros(size=(len(entity_indices[source]), len(entity_indices[target])), dtype=bool)
        for s, ts in edges[rel_type].items():
            for t in ts:
                ss = entity_indices[source].index(s)
                tt = entity_indices[target].index(t)
                adjacencies[rel_type][ss, tt] = True
    logging.debug("Dataset has %d entities", data.num_entities)
    try:
        for batch_num, batch_ids in batch_to_batch_ids.items():
            representation_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            bottleneck_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            entities = [data.entity(i) for (_, i) in batch_ids]
            entity_type_to_batch_indices, entity_type_to_property_indices, packed_properties = model.pack_properties(
                entities
            )
            encoded_properties = model.encode_properties(packed_properties)
            encoded_entities = model.create_autoencoder_inputs(
                encoded_properties,
                entity_type_to_batch_indices,
                entity_type_to_property_indices             
            )
            bottlenecks, autoencoder_output = model.run_autoencoder(
                0,
                encoded_entities
            )
            torch.save((entity_type_to_batch_indices, entity_type_to_property_indices, [autoencoder_output]), representation_storage[batch_num])
            torch.save([bottlenecks], bottleneck_storage[batch_num])
        for depth in range(1, model.depth + 1):
            prev_bottleneck_list = []
            ets = set()
            for batch_num in batch_to_batch_ids.keys():
                prev_bottleneck_list.append(torch.load(bottleneck_storage[batch_num])[-1])
                for k in prev_bottleneck_list[0].keys():
                    ets.add(k)
            prev_bottlenecks = {k : torch.cat([x[k] for x in prev_bottleneck_list], 0) for k in ets}
            for batch_num, batch_ids in batch_to_batch_ids.items():
                entity_type_to_batch_indices, entity_type_to_property_indices, prev_ae_outputs = torch.load(representation_storage[batch_num])
                for rel_name, rel_spec in data.schema["relationships"].items():
                    s_et = rel_spec["source_entity_type"]
                    t_et = rel_spec["target_entity_type"]
                    num_batch_src = len(entity_type_to_batch_indices[s_et])
                    num_batch_tgt = len(entity_type_to_batch_indices[t_et])
                    num_src_total = len(data.get_type_ids(s_et))
                    num_tgt_total = len(data.get_type_ids(t_et))
                adjacency_mappings = {et : [] for et in data.schema["entity_types"].keys()}
                for _, bid in batch_ids:
                    et, ep = data.get_entity_type_and_position(bid)
                    adjacency_mappings[et].append(ep)
                encoded_entities = model.create_structured_autoencoder_inputs(
                    depth,
                    prev_ae_outputs[-1],
                    prev_bottlenecks,
                    adjacencies,
                    adjacency_mappings
                )
                bottlenecks, autoencoder_output = model.run_autoencoder(
                    depth,
                    encoded_entities
                )
                
                torch.save((entity_type_to_batch_indices, entity_type_to_property_indices, prev_ae_outputs + [autoencoder_output]), representation_storage[batch_num])
                torch.save(torch.load(bottleneck_storage[batch_num]) + [bottlenecks], bottleneck_storage[batch_num])

        for batch_num, batch_ids in batch_to_batch_ids.items():
            entity_type_to_batch_indices, entity_type_to_property_indices, prev_ae_outputs = torch.load(representation_storage[batch_num])
            prev_bottleneck_list = []
            ets = set()
            for batch_num in batch_to_batch_ids.keys():
                prev_bottleneck_list.append(torch.load(bottleneck_storage[batch_num])[-1])
                for k in prev_bottleneck_list[0].keys():
                    ets.add(k)
            all_bottlenecks = {k : torch.cat([x[k] for x in prev_bottleneck_list], 0) for k in ets}
            all_ae_outputs = {k : torch.cat([p[k] for p in prev_ae_outputs], 1) for k in prev_ae_outputs[0].keys()}
            resized_encoded_entities = model.project_autoencoder_outputs(all_ae_outputs)
            indices, decoded_properties = model.decode_properties(
                resized_encoded_entities,
                entity_type_to_batch_indices,
                entity_type_to_property_indices
            )
            entities = [data.entity(i) for _, i in batch_ids]
            reconstructions_by_id = model.reconstruct_entities(            
                decoded_properties,
                entities,
                entity_type_to_batch_indices,
            )
            bottlenecks_by_id = model.assemble_bottlenecks(
                {k : v for k, v in all_bottlenecks.items()},
                entities,
                entity_type_to_batch_indices
            )
            yield (entities, 0.0, reconstructions_by_id, bottlenecks_by_id)
    except Exception as e:
        raise e
    finally:
        for tfname in list(bottleneck_storage.values()) + list(representation_storage.values()):
            try:
                os.remove(tfname)
            except Exception as e:
                logging.info("Could not clean up temporary file: %s", tfname)
                raise e
