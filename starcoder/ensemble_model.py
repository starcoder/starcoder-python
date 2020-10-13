import pickle
import re
import sys
import argparse
import torch
import json
import numpy
import scipy.sparse
import gzip
from torch.utils.data import DataLoader, Dataset
import functools
import numpy

import logging
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from starcoder.random import random
from starcoder.summarizer import SingleSummarizer
from starcoder.autoencoder import BasicAutoencoder
from starcoder.projector import MLPProjector
from starcoder.activation import Activation
from starcoder.registry import field_model_classes, summarizer_classes, projector_classes
from starcoder.adjacency import Adjacencies, Adjacency
from starcoder.entity import UnpackedEntity, PackedEntity, StackedEntities, stack_entities, unstack_entities
from typing import List, Dict, Any, Type, Optional, Union, cast, Tuple
from starcoder.schema import Schema
from torch import Tensor
from starcoder.base import StarcoderObject

logger = logging.getLogger(__name__)

class Ensemble(StarcoderObject, torch.nn.Module): # type: ignore[type-arg]
    def __init__(self) -> None:
        super(Ensemble, self).__init__()

class GraphAutoencoder(Ensemble):
    def __init__(self,
                 schema: Schema,
                 depth: int,
                 autoencoder_shapes: List[int],
                 reverse_relationships: bool=False,
                 summarizers: Type[SingleSummarizer]=SingleSummarizer,
                 activation: Activation=torch.nn.functional.relu,
                 projected_size: Optional[int]=None,
                 base_entity_representation_size: int=8,
                 device: Any=torch.device("cpu")) -> None:
        """
        """
        super(GraphAutoencoder, self).__init__()
        self.reverse_relationships = reverse_relationships
        self.schema = schema
        self.depth = depth
        self.device = device
        self.base_entity_representation_size = base_entity_representation_size        
        self.autoencoder_shapes = autoencoder_shapes
        self.bottleneck_size = 0 if autoencoder_shapes in [[], None] else autoencoder_shapes[-1]
        
        # An encoder for each field that turns its data type into a fixed-size representation
        field_encoders = {}
        for field_name, field_object in self.schema.data_fields.items():
            field_type = type(field_object)
            if field_type not in field_model_classes:
                raise Exception("There is no encoder architecture registered for field type '{}'".format(field_type))
            field_encoders[field_name] = field_model_classes[field_type][0](field_object, activation)
        self.field_encoders = torch.nn.ModuleDict(field_encoders)

        # The size of an encoded entity is the sum of the base representation size, the encoded
        # sizes of its possible fields, and a binary indicator for the presence of each field.
        self.boundary_sizes = {}
        for entity_type in self.schema.entity_types.values():
            self.boundary_sizes[entity_type.name] = self.base_entity_representation_size
            for field_name in entity_type.data_fields:
                self.boundary_sizes[entity_type.name] += self.field_encoders[field_name].output_size + 1 # type: ignore

        # An autoencoder for each entity type and depth
        # The first has input/output layers of size equal to the size of the corresponding entity's representation size
        # The rest have input/output layers of that size plus a bottleneck size for each possible (normal or reverse) relationship
        entity_autoencoders = {}
        for entity_type in self.schema.entity_types.values():
            boundary_size = self.boundary_sizes[entity_type.name]
            entity_type_autoencoders = [BasicAutoencoder([boundary_size] + self.autoencoder_shapes, activation)]
            for _ in entity_type.relationship_fields:
                boundary_size += self.bottleneck_size
            if self.reverse_relationships:
                for _ in entity_type.reverse_relationship_fields:
                    boundary_size += self.bottleneck_size
            for depth in range(self.depth):
                entity_type_autoencoders.append(BasicAutoencoder([boundary_size] + self.autoencoder_shapes, activation))
            entity_autoencoders[entity_type.name] = torch.nn.ModuleList(entity_type_autoencoders)
        self.entity_autoencoders = torch.nn.ModuleDict(entity_autoencoders)

        # A summarizer for each relationship particant (source or target), to reduce one-to-many relationships to a fixed size
        if self.depth > 0:
            relationship_source_summarizers = {}
            relationship_target_summarizers = {}
            for relationship_field in self.schema.relationship_fields.values():
                relationship_source_summarizers[relationship_field.name] = summarizers(self.bottleneck_size, activation)
                relationship_target_summarizers[relationship_field.name] = summarizers(self.bottleneck_size, activation)
            self.relationship_source_summarizers = torch.nn.ModuleDict(relationship_source_summarizers)
            self.relationship_target_summarizers = torch.nn.ModuleDict(relationship_target_summarizers)

        # MLP for each entity type to project representations to a common size
        # note the largest boundary size
        self.projected_size = cast(int, projected_size if projected_size != None else max(self.boundary_sizes.values()))
        projectors = {}
        for entity_type in self.schema.entity_types.values():
            boundary_size = self.boundary_sizes.get(entity_type.name, 0)
            if self.depth > 0:
                for _ in entity_type.relationship_fields:
                    boundary_size += self.bottleneck_size
                if self.reverse_relationships:
                    for _ in entity_type.reverse_relationship_fields:
                        boundary_size += self.bottleneck_size
            projectors[entity_type.name] = MLPProjector(boundary_size, self.projected_size, activation)
        self.projectors = torch.nn.ModuleDict(projectors)
        
        # A decoder for each field that takes a projected representation and generates a value of the field's data type
        field_decoders = {}
        self.field_losses = {}
        for field_name, field_object in self.schema.data_fields.items():
            field_type = type(field_object)
            field_decoders[field_name] = field_model_classes[field_type][1](field_object,
                                                                            self.projected_size,
                                                                            activation)
            self.field_losses[field_name] = field_model_classes[field_type][2](field_object)
        self.field_decoders = torch.nn.ModuleDict(field_decoders)

    def cuda(self, device: Any="cuda:0") -> "GraphAutoencoder":
        self.device = torch.device(device)
        return super(GraphAutoencoder, self).cuda()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_fields(self, entities: StackedEntities) -> Dict[str, Tensor]:
        logger.debug("Encoding each input field to a fixed-length representation")
        retval = {k : self.field_encoders[k](v) if k in self.field_encoders else v for k, v in entities.items()}
        return retval
    
    def create_autoencoder_inputs(self, encoded_fields: Dict[str, Tensor], entity_indices: Dict[str, Tensor]) -> Dict[str, Tensor]:
        logger.debug("Constructing entity-autoencoder inputs by concatenating field encodings")
        autoencoder_inputs = {}
        autoencoder_input_lists = {}
        for entity_type in self.schema.entity_types.values():
            #
            # each appended value should have shape (entity_count x encoding_width)
            #            
            autoencoder_input_lists[entity_type.name] = [torch.zeros(size=(entity_indices[entity_type.name].shape[0], 8), dtype=torch.float32, device=self.device)]
            for field_name in entity_type.data_fields:
                vals = torch.index_select(
                    encoded_fields[field_name], 
                    0, 
                    entity_indices[entity_type.name]
                )
                autoencoder_input_lists[entity_type.name].append(vals)
                #print(field_name, vals.shape)
                inds = torch.where(
                    torch.isnan(vals[:, 0]), 
                    torch.zeros(size=(vals.shape[0], 1), device=self.device), 
                    torch.ones(size=(vals.shape[0], 1), device=self.device)
                )
                autoencoder_input_lists[entity_type.name].append(
                    torch.zeros(
                        size=(
                            autoencoder_input_lists[entity_type.name][-1].shape[0], 
                            1
                        ),
                        device=self.device
                    )
                )
            autoencoder_input_lists[entity_type.name].append(
                torch.zeros(
                    size=(entity_indices[entity_type.name].shape[0], 0), 
                    dtype=torch.float32,
                    device=self.device
                )
            )
            autoencoder_inputs[entity_type.name] = torch.cat(
                autoencoder_input_lists[entity_type.name], 
                1
            )
            autoencoder_inputs[entity_type.name][torch.isnan(autoencoder_inputs[entity_type.name])] = 0
        logger.debug("Shapes: %s", {k : v.shape for k, v in autoencoder_inputs.items()})
        return autoencoder_inputs    

    def run_first_autoencoder_layer(self, autoencoder_inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        logging.debug("Starting first (graph-unaware) autoencoding layer")
        bottlenecks = {}
        autoencoder_outputs = {}
        for entity_type in self.schema.entity_types.values():
            #print(entity_type)
            entity_reps = autoencoder_inputs.get(entity_type.name, None)
            #print(entity_reps)
            if entity_reps != None:
                entity_outputs, bns, losses = self.entity_autoencoders[entity_type.name][0](autoencoder_inputs[entity_type.name]) # type: ignore
                if entity_outputs != None:
                    autoencoder_outputs[entity_type.name] = entity_outputs
                if bns != None:
                    bottlenecks[entity_type.name] = bns
        logging.debug("Finished first (graph-unaware) autoencoding layer")
        return (bottlenecks, autoencoder_outputs)

    def run_structured_autoencoder_layer(self,
                                         depth: int,
                                         autoencoder_inputs: Dict[str, Tensor],
                                         prev_bottlenecks: Dict[str, Tensor],
                                         adjacencies: Adjacencies,
                                         batch_index_to_entity_index: Tensor,
                                         entity_indices: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        logging.debug("Starting graph-aware autoencoding layer number %d", depth)
        bottlenecks = {}
        autoencoder_outputs = {}
        num_all_entities = sum([len(x) for x in entity_indices.values()])
        index_space = torch.arange(0, num_all_entities, 1, device=self.device)
        for entity_type_name, entity_type_inputs in autoencoder_inputs.items(): #self.schema.entity_types.values():
            entity_type = self.schema.entity_types[entity_type_name]
            autoencoder_outputs[entity_type_name] = entity_type_inputs.narrow(1, 0, self.entity_autoencoders[entity_type_name][0].output_size) # type: ignore
            num_entities = entity_type_inputs.shape[0]            
            other_rep_list = []
            for rel_name in entity_type.relationship_fields + (entity_type.reverse_relationship_fields if self.reverse_relationships else []):
                summarize = self.relationship_target_summarizers[rel_name]
                rel_field = self.schema.relationship_fields[rel_name]
                target_entity_type = rel_field.target_entity_type
                rep_size = self.entity_autoencoders[target_entity_type][depth - 1].bottleneck_size # type: ignore
                relationship_reps = torch.zeros(size=(num_entities, rep_size), device=self.device)
                for i, index in enumerate(entity_indices[entity_type_name]):
                    #related_indices: List[int] = []
                    if rel_name in adjacencies:
                        related_indices_ = index_space.masked_select(adjacencies[rel_name][index])
                        #print(batch_index_to_entity_index)
                        related_indices = torch.tensor([batch_index_to_entity_index[j] for j in related_indices_], device=self.device)
                        if len(related_indices) > 0:
                            try:
                                obns = torch.index_select(
                                    prev_bottlenecks[target_entity_type], 
                                    0, 
                                    related_indices
                                )
                            except Exception as e:
                                print(related_indices, prev_bottlenecks[target_entity_type].shape, batch_index_to_entity_index)
                                raise e
                            relationship_reps[i] = summarize(obns)
                other_rep_list.append(relationship_reps)

            sh = list(autoencoder_outputs[entity_type.name].shape)
            sh[1] = 0
            other_reps = torch.cat(other_rep_list, 1) if len(other_rep_list) > 0 else torch.zeros(size=tuple(sh), device=self.device)
            autoencoder_inputs[entity_type.name] = torch.cat([autoencoder_outputs[entity_type.name], other_reps], 1)
            if depth > len(self.entity_autoencoders[entity_type.name]) - 1: # type: ignore
                raise Exception()
                logger.debug("At depth %d, while the model was trained for depth %d, so reusing final autoencoder",
                             depth + 1,
                             len(self._entity_autoencoders[entity_type.name]))
                entity_outputs, bns, losses = self.entity_autoencoders[entity_type.name][-1](autoencoder_inputs[entity_type.name])
            else:
                entity_outputs, bns, losses = self.entity_autoencoders[entity_type.name][depth](autoencoder_inputs[entity_type.name]) # type: ignore
            autoencoder_outputs[entity_type.name] = entity_outputs
            #if entity_outputs.shape[1] != 0:
            #   bottlenecks[entity_indices[entity_type.name]] = bns
            
            entity_reps = autoencoder_inputs.get(entity_type.name, None)
            if entity_reps != None:
                entity_outputs, bns, losses = self.entity_autoencoders[entity_type.name][depth](autoencoder_inputs[entity_type.name]) # type: ignore
                if entity_outputs != None:
                    autoencoder_outputs[entity_type.name] = entity_outputs
                if bns != None:
                    bottlenecks[entity_type.name] = bns
        logging.debug("Finished graph-aware autoencoding layer number %d", depth)

        return (bottlenecks, autoencoder_outputs)    
                    
    def project_autoencoder_outputs(self, autoencoder_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        resized_autoencoder_outputs = {}
        for entity_type_name, ae_output in autoencoder_outputs.items():
            resized_autoencoder_outputs[entity_type_name] = self.projectors[entity_type_name](ae_output)                
        return resized_autoencoder_outputs
    
    def decode_fields(self, resized_encoded_entities: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        """
        """
        reconstructions: Dict[str, Any] = {}        
        for field in self.schema.data_fields.values():
            for entity_type_name, v in resized_encoded_entities.items():
                reconstructions[entity_type_name] = reconstructions.get(entity_type_name, {})
                reconstructions[entity_type_name][field.name] = self.field_decoders[field.name](v)
        return reconstructions
    
    def assemble_entities(self, decoded_fields: Dict[str, Dict[str, Tensor]], entity_indices: Dict[str, Tensor]) -> Dict[str, Tensor]:
       num = sum([len(v) for v in entity_indices.values()])
       retval: Dict[str, Tensor] = {}
       for entity_type_name, fields in decoded_fields.items():
           indices = entity_indices[entity_type_name]
           for field_name, field_values in fields.items():
               retval[field_name] = retval.get(
                   field_name, 
                   torch.empty(
                       size=(num,) + field_values.shape[1:],
                       device=self.device,
                   )
               )
               retval[field_name][indices] = field_values
       return retval
    
    def compute_indices(self, entities: StackedEntities) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[Tuple[str, str], Tensor]]:
        entity_indices = {}
        entity_masks = {}
        field_indices = {}
        field_masks = {}
        entity_field_indices = {}
        entity_field_masks = {}
        logger.debug("Assembling entity, field, and (entity, field) indices")        
        index_space = torch.arange(0, entities[self.schema.entity_type_field.name].shape[0], 1, device=self.device)
        for field in self.schema.data_fields.values():
            # FIXME: hack for RNNs            
            if field.type_name in ["sequential", "text"]:
                if entities[field.name].shape[1] == 0:
                    field_masks[field.name] = torch.full((entities[field.name].shape[0],), False, device=self.device, dtype=torch.bool)
                else:
                    field_masks[field.name] = entities[field.name][:, 0] != 0
            else:
                field_masks[field.name] = ~torch.isnan(torch.reshape(entities[field.name], (entities[field.name].shape[0], -1)).sum(1))
            field_indices[field.name] = index_space.masked_select(field_masks[field.name])
            
        for entity_type in self.schema.entity_types.values():
            entity_masks[entity_type.name] = torch.tensor((entities[self.schema.entity_type_field.name] == entity_type.name), device=self.device)
            entity_indices[entity_type.name] = index_space.masked_select(entity_masks[entity_type.name])
            for field_name in entity_type.data_fields:
                entity_field_masks[(entity_type.name, field_name)] = entity_masks[entity_type.name] & field_masks[field_name]
                entity_field_indices[(entity_type.name, field_name)] = index_space.masked_select(entity_field_masks[(entity_type.name, field_name)])
        return (entity_indices, field_indices, entity_field_indices)
    
    def forward(self, entities: StackedEntities, adjacencies: Adjacencies) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        logger.debug("Starting forward pass")
        entity_indices, field_indices, entity_field_indices = self.compute_indices(entities)
        batch_index_to_entity_index = torch.tensor([i for _, i in sorted(sum([[(bi.item(), ei) for ei, bi in enumerate(ev)] for k, ev in entity_indices.items()], []))], device=self.device)
        num_entities = len(entities[self.schema.id_field.name])
        rev_adjacencies = {k : v.T for k, v in adjacencies.items()}
        field_encodings = self.encode_fields(entities)
        encoded_entities = self.create_autoencoder_inputs(field_encodings, entity_indices)
        bottlenecks, encoded_entities = self.run_first_autoencoder_layer(encoded_entities)
        for depth in range(1, self.depth + 1):
            bottlenecks, encoded_entities = self.run_structured_autoencoder_layer(depth,                
                                                                                  encoded_entities,
                                                                                  prev_bottlenecks=bottlenecks,
                                                                                  adjacencies=adjacencies,
                                                                                  batch_index_to_entity_index=batch_index_to_entity_index,
                                                                                  entity_indices=entity_indices)
        resized_encoded_entities = self.project_autoencoder_outputs(encoded_entities)
        decoded_fields = self.decode_fields(resized_encoded_entities)
        decoded_entities = self.assemble_entities(decoded_fields, entity_indices)
        normalized_entities = {k : self.field_decoders[k].normalize(v.cpu().detach()) for k, v in decoded_entities.items()}
        for k, v in entities.items():
            if k not in decoded_entities:
                decoded_entities[k] = v
                normalized_entities[k] = v
        bottlenecks_by_id = {}
        for entity_type_name, bns in bottlenecks.items():
            #print(entities[self.schema.id_field.name])
            ids = entities[self.schema.id_field.name][entity_indices[entity_type_name].cpu()]
            # strange how numpy.array can be a scalar here!
            if isinstance(ids, str):
                ids = [ids]
            for i, bn in zip(ids, bns):
                bottlenecks_by_id[i] = bn
        logger.debug("Returning reconstructions, bottlenecks, and autoencoder I/O pairs")
        return (decoded_entities, normalized_entities, bottlenecks_by_id)

    # Recursively initialize model weights
    def init_weights(m: Any) -> None:
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv1d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
