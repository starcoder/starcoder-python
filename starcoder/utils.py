from typing import Dict, List, Any, Tuple, Generator
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.schema import Schema
from starcoder.dataset import Dataset
from starcoder.entity import stack_entities, UnpackedEntity, PackedEntity, Index, ID
from starcoder.adjacency import Adjacencies
import torch
import logging
import tempfile
import json
import gzip
import os

logger = logging.getLogger("starcoder")

def simple_loss_policy(losses_by_field: Dict[str, List[float]]) -> float:
    loss_by_field = {k : sum(v) for k, v in losses_by_field.items()}
    retval = sum(loss_by_field.values())
    return retval

def compute_losses(model: GraphAutoencoder,
                   entities: Dict[str, Any],
                   reconstructions: Dict[str, Any],
                   schema: Schema) -> Dict[str, torch.Tensor]:
    """
    Gather and return all the field losses as a nested dictionary where the first
    level of keys are field types, and the second level are field names.
    """
    logger.debug("Started computing all losses")
    losses = {}
    for entity_type_name, entity_type in schema.entity_types.items():
        for field_name in entity_type.data_fields:
            field_values = entities[field_name]
            #for field_name, field_values in entities.items():
            #if field_name in schema.data_fields and field_name in reconstructions:
            field = schema.data_fields[field_name]
            logger.debug("Computing losses for field %s of type %s", field.name, field.type_name)
            reconstruction_values = reconstructions[field.name]
            field_losses = model.field_losses[field.name](reconstruction_values, field_values)
            mask = ~torch.isnan(field_losses)
            losses[field.name] = torch.masked_select(field_losses, mask)
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
                        subselect: bool=False,
                        strict: bool=True,
                        mask_tests: Any=[]) -> Tuple[Any, Dict[Any, Any]]:
    old_mode = model.training
    model.train(train)
    loss_by_field: Dict[str, Any] = {}
    loss = 0.0
    for batch_num, (full_entities, full_adjacencies) in enumerate(batchifier(data, batch_size)):
        logger.debug("Processing batch #%d", batch_num)
        batch_loss_by_field = {}
        if gpu:
            full_entities = {k : v.cuda() if hasattr(v, "cuda") else v for k, v in full_entities.items()}
            full_adjacencies = {k : v.cuda() for k, v in full_adjacencies.items()}
        optim.zero_grad()
        reconstructions, norm, bottlenecks = model(full_entities, full_adjacencies)
        for field, losses in compute_losses(model, full_entities, reconstructions, data.schema).items():
            if losses.shape[0] > 0:
                batch_loss_by_field[field] = losses
        logging.debug("Applying loss policy")
        batch_loss = loss_policy(batch_loss_by_field)
        loss += batch_loss.clone().detach()
        if train:            
            logging.debug("Back-propagating")
            batch_loss.backward()
            logging.debug("Stepping")
            optim.step()
        logging.debug("Assembling losses by field")
        for field, v in batch_loss_by_field.items():
            loss_by_field[field] = loss_by_field.get(field, [])
            loss_by_field[field].append(v.clone().detach())
        logging.debug("Finished batch #%d", batch_num)
        #sys.exit()
    model.train(old_mode)
    return (loss, loss_by_field)

def apply_to_components(model: GraphAutoencoder,
                        batchifier: Any,
                        data: Dataset,
                        batch_size: int,
                        gpu: bool) -> Generator[Tuple[Any, Dict[Any, Any]], None, None]:
    old_mode = model.training
    model.train(False)
    for batch_num, (full_entities, full_adjacencies) in enumerate(batchifier(data, batch_size)):
        logger.debug("Processing batch #%d", batch_num)
        #batch_loss_by_field = {}
        if gpu:
            full_entities = {k : v.cuda() if hasattr(v, "cuda") else v for k, v in full_entities.items()}
            full_adjacencies = {k : v.cuda() for k, v in full_adjacencies.items()}
        yield model(full_entities, full_adjacencies)
    model.train(old_mode)
    #return (reconstructions, bottlenecks)


def run_epoch(model: GraphAutoencoder,
              batchifier: Any,
              optimizer: Any,
              loss_policy: Any,
              train_data: Dataset,
              dev_data: Dataset,
              batch_size: int,
              gpu: bool,
              mask_tests: Any=[],
              subselect: bool=False) -> Tuple[Any, Any, Any, Any]:
    model.train(True)
    logger.debug("Running over training data")
    train_loss, train_loss_by_field = run_over_components(model,
                                                          batchifier,
                                                          optimizer, 
                                                          loss_policy,
                                                          train_data, 
                                                          batch_size, 
                                                          gpu,
                                                          subselect=subselect,
                                                          train=True,
                                                          mask_tests=mask_tests,
    )
    logger.debug("Running over dev data")
    model.train(False)
    dev_loss, dev_loss_by_field = run_over_components(model,
                                                      batchifier,
                                                      optimizer, 
                                                      loss_policy,
                                                      dev_data, 
                                                      batch_size, 
                                                      gpu,
                                                      subselect=subselect,
                                                      train=False,
                                                      mask_tests=mask_tests,
    )

    return (train_loss.clone().detach().cpu(),
            #train_loss_by_field,
            {k : sum([x.sum() for x in v]) for k, v in train_loss_by_field.items()},
            #{k : sum([x.sum() for x in v]) for k, v in dev_loss_by_field.items()},
            #{k : [v.clone().detach().cpu() for v in vv] for k, vv in train_loss_by_field.items()},
            dev_loss.clone().detach().cpu(),
            {k : sum([x.sum() for x in v]) for k, v in dev_loss_by_field.items()},
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
            stacked_batch_entities = stack_entities(packed_batch_entities, data.schema.data_fields)
            encoded_batch_entities = model.encode_fields(stacked_batch_entities)
            entity_indices, field_indices, entity_field_indices = model.compute_indices(stacked_batch_entities)            
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
            decoded_fields = model.decode_fields(model.project_autoencoder_outputs(outputs))
            decoded_entities = model.assemble_entities(decoded_fields, entity_indices)
            decoded_fields = {k : {kk : vv for kk, vv in v.items() if kk in data.schema.entity_types[k].data_fields} for k, v in decoded_fields.items()}
            ordered_bottlenecks = {}
            for entity_type, indices in entity_indices.items():
                for i, index in enumerate(indices):
                    ordered_bottlenecks[index.item()] = bottlenecks[entity_type][i]

            for i, eid in enumerate(b_ids):
                original_entity = data.entity(eid)                 
                entity_type_name = original_entity[data.schema.entity_type_field.name]
                entity_data_fields = data.schema.entity_types[entity_type_name].data_fields

                reconstructed_entity = {data.schema.entity_type_field.name : entity_type_name}
                for field_name in original_entity.keys():
                    if field_name in entity_data_fields:
                        reconstructed_entity[field_name] = decoded_entities[field_name][i].tolist()
                reconstructed_entity = data.schema.unpack(reconstructed_entity)
                entity = {"original" : original_entity,
                          # include missing-but-generated fields
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
            stacked_batch_entities = stack_entities(packed_batch_entities, data.schema.data_fields)
            encoded_batch_entities = model.encode_fields(stacked_batch_entities)
            entity_indices, field_indices, entity_field_indices = model.compute_indices(stacked_batch_entities)            
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
            decoded_fields = model.decode_fields(model.project_autoencoder_outputs(outputs))
            decoded_entities = model.assemble_entities(decoded_fields, entity_indices)
            decoded_fields = {k : {kk : vv for kk, vv in v.items() if kk in data.schema.entity_types[k].data_fields} for k, v in decoded_fields.items()}
            ordered_bottlenecks = {}
            for entity_type, indices in entity_indices.items():
                for i, index in enumerate(indices):
                    ordered_bottlenecks[index.item()] = bottlenecks[entity_type][i]

            for i, eid in enumerate(b_ids):
                original_entity = data.entity(eid)                 
                entity_type_name = original_entity[data.schema.entity_type_field.name]
                entity_data_fields = data.schema.entity_types[entity_type_name].data_fields

                reconstructed_entity = {data.schema.entity_type_field.name : entity_type_name}
                for field_name in original_entity.keys():
                    if field_name in entity_data_fields:
                        reconstructed_entity[field_name] = decoded_entities[field_name][i].tolist()
                reconstructed_entity = data.schema.unpack(reconstructed_entity)
                entity = {"original" : original_entity,
                          # include missing-but-generated fields
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
