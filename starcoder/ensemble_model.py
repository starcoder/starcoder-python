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
from torch.nn import Dropout
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
from starcoder.registry import property_model_classes, summarizer_classes, projector_classes
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
                 depthwise_boost="none",
                 device: Any=torch.device("cpu"),
                 train_neuron_dropout=0.0) -> None:
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
        self.dropout = Dropout(train_neuron_dropout)
        self.depthwise_boost = depthwise_boost
        
        # An encoder for each property that turns its data type into a fixed-size representation
        property_encoders = {}
        for property_name, property_object in self.schema.properties.items():
            property_type = type(property_object)
            if property_type not in property_model_classes:
                raise Exception("There is no encoder architecture registered for property type '{}'".format(property_type))
            property_encoders[property_name] = property_model_classes[property_type][0](property_object, activation)
        self.property_encoders = torch.nn.ModuleDict(property_encoders)

        # The size of an encoded entity is the sum of the base representation size, the encoded
        # sizes of its possible properties, and a binary indicator for the presence of each property.
        self.boundary_sizes = {}
        for entity_type in self.schema.entity_types.values():
            self.boundary_sizes[entity_type.name] = self.base_entity_representation_size
            for property_name in entity_type.properties:
                self.boundary_sizes[entity_type.name] += self.property_encoders[property_name].output_size + 1 # type: ignore

        # An autoencoder for each entity type and depth
        # The first has input/output layers of size equal to the size of the corresponding entity's representation size
        # The rest have input/output layers of that size plus a bottleneck size for each possible (normal or reverse) relationship
        entity_autoencoders = {}
        for entity_type in self.schema.entity_types.values():
            boundary_size = self.boundary_sizes[entity_type.name]
            entity_type_autoencoders = [BasicAutoencoder([boundary_size] + self.autoencoder_shapes, activation)]
            for _ in entity_type.relationships:
                boundary_size += self.bottleneck_size
            if self.reverse_relationships:
                for _ in entity_type.reverse_relationships:
                    boundary_size += self.bottleneck_size
            for depth in range(self.depth):
                entity_type_autoencoders.append(BasicAutoencoder([boundary_size] + self.autoencoder_shapes, activation))
            entity_autoencoders[entity_type.name] = torch.nn.ModuleList(entity_type_autoencoders)
        self.entity_autoencoders = torch.nn.ModuleDict(entity_autoencoders)

        # A summarizer for each relationship particant (source or target), to reduce one-to-many relationships to a fixed size
        if self.depth > 0:
            relationship_source_summarizers = {}
            relationship_target_summarizers = {}
            for relationship in self.schema.relationships.values():
                relationship_source_summarizers[relationship.name] = summarizers(self.bottleneck_size, activation)
                relationship_target_summarizers[relationship.name] = summarizers(self.bottleneck_size, activation)
            self.relationship_source_summarizers = torch.nn.ModuleDict(relationship_source_summarizers)
            self.relationship_target_summarizers = torch.nn.ModuleDict(relationship_target_summarizers)

        # MLP for each entity type to project representations to a common size
        self.projected_size = cast(int, projected_size if projected_size != None else max(self.boundary_sizes.values()))
        projectors = {}
        for entity_type in self.schema.entity_types.values():
            boundary_size = self.boundary_sizes.get(entity_type.name, 0)
            if self.depth > 0:
                for _ in entity_type.relationships:
                    boundary_size += self.bottleneck_size
                if self.reverse_relationships:
                    for _ in entity_type.reverse_relationships:
                        boundary_size += self.bottleneck_size
            projectors[entity_type.name] = MLPProjector(boundary_size, self.projected_size, activation)
        self.projectors = torch.nn.ModuleDict(projectors)
        
        # A decoder for each property that takes a projected representation and generates a value of the property's data type
        property_decoders = {}
        for property_name, property_object in self.schema.properties.items():
            property_type = type(property_object)
            input_size = self.projected_size #self.projected_size if self.depthwise_boost == "none" else self.projected_size * (self.depth + 1)
            property_decoders[property_name] = property_model_classes[property_type][1](
                property_object,
                input_size,
                activation,
                encoder=self.property_encoders[property_name]
            )
        self.property_decoders = torch.nn.ModuleDict(property_decoders)

        # A way to calculate loss for each property
        self.property_losses = {}
        for property_name, property_object in self.schema.properties.items():
            property_type = type(property_object)            
            self.property_losses[property_name] = property_model_classes[property_type][2](property_object)            

    def cuda(self, device: Any="cuda:0") -> "GraphAutoencoder":
        self.device = torch.device(device)
        return super(GraphAutoencoder, self).cuda()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_properties(self, entities: StackedEntities) -> Dict[str, Tensor]:
        logger.debug("Encoding each input property to a fixed-length representation")
        #print(entities["tweet_text"].sum(1))
        retval = {k : self.property_encoders[k](v) if k in self.property_encoders else v for k, v in entities.items()}
        return retval
    
    def create_autoencoder_inputs(self, encoded_properties: Dict[str, Tensor], entity_indices: Dict[str, Tensor]) -> Dict[str, Tensor]:
        logger.debug("Constructing entity-autoencoder inputs by concatenating property encodings")
        autoencoder_inputs = {}
        autoencoder_input_lists = {}
        for entity_type in self.schema.entity_types.values():
            #
            # each appended value should have shape (entity_count x encoding_width)
            #            
            autoencoder_input_lists[entity_type.name] = [torch.zeros(size=(entity_indices[entity_type.name].shape[0], 8), dtype=torch.float32, device=self.device)]
            for property_name in entity_type.properties:
                vals = torch.index_select(
                    encoded_properties[property_name], 
                    0, 
                    entity_indices[entity_type.name]
                )
                autoencoder_input_lists[entity_type.name].append(vals)
                #print(property_name, vals.shape)
                nan_test = vals.reshape(vals.shape[0], -1 if vals.shape[0] != 0 else 0).sum(1)
                #print(vals.shape, nan_test.shape)
                inds = torch.where(
                    torch.isnan(nan_test), #vals[:, 0]), 
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
        autoencoder_inputs = {k : self.dropout(v) for k, v in autoencoder_inputs.items()}
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
            for rel_name in entity_type.relationships + (entity_type.reverse_relationships if self.reverse_relationships else []):
                summarize = self.relationship_target_summarizers[rel_name]
                relationship = self.schema.relationships[rel_name]
                target_entity_type = relationship.target_entity_type
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
    
    def decode_properties(self, resized_encoded_entities: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        """
        """
        reconstructions: Dict[str, Any] = {}        
        for property in self.schema.properties.values():
            for entity_type_name, v in resized_encoded_entities.items():
                reconstructions[entity_type_name] = reconstructions.get(entity_type_name, {})
                reconstructions[entity_type_name][property.name] = self.property_decoders[property.name](v)
        return reconstructions
    
    def assemble_entities(self, decoded_properties: Dict[str, Dict[str, Tensor]], entity_indices: Dict[str, Tensor]) -> Dict[str, Tensor]:
       num = sum([len(v) for v in entity_indices.values()])
       retval: Dict[str, Tensor] = {}
       for entity_type_name, properties in decoded_properties.items():
           indices = entity_indices[entity_type_name]
           for property_name, property_values in properties.items():
               retval[property_name] = retval.get(
                   property_name, 
                   torch.empty(
                       size=(num,) + property_values.shape[1:],
                       device=self.device,
                   )
               )
               retval[property_name][indices] = property_values
       return retval
    
    def compute_indices(self, entities: StackedEntities) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[Tuple[str, str], Tensor]]:
        entity_indices = {}
        entity_masks = {}
        property_indices = {}
        property_masks = {}
        entity_property_indices = {}
        entity_property_masks = {}
        logger.debug("Assembling entity, property, and (entity, property) indices")        
        index_space = torch.arange(0, entities[self.schema.entity_type_property.name].shape[0], 1, device=self.device)
        for property in self.schema.properties.values():
            # FIXME: hack for RNNs            
            if property.type_name in ["sequential", "text"]:
                if entities[property.name].shape[1] == 0:
                    property_masks[property.name] = torch.full((entities[property.name].shape[0],), False, device=self.device, dtype=torch.bool)
                else:
                    property_masks[property.name] = entities[property.name][:, 0] != 0
            else:
                property_masks[property.name] = ~torch.isnan(torch.reshape(entities[property.name], (entities[property.name].shape[0], -1)).sum(1))
            property_indices[property.name] = index_space.masked_select(property_masks[property.name])
            
        for entity_type in self.schema.entity_types.values():
            entity_masks[entity_type.name] = torch.tensor((entities[self.schema.entity_type_property.name] == entity_type.name), device=self.device)
            entity_indices[entity_type.name] = index_space.masked_select(entity_masks[entity_type.name])
            for property_name in entity_type.properties:
                entity_property_masks[(entity_type.name, property_name)] = entity_masks[entity_type.name] & property_masks[property_name]
                entity_property_indices[(entity_type.name, property_name)] = index_space.masked_select(entity_property_masks[(entity_type.name, property_name)])
        return (entity_indices, property_indices, entity_property_indices)
    
    def forward(self, entities: StackedEntities, adjacencies: Adjacencies) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        logger.debug("Starting forward pass")
        entity_indices, property_indices, entity_property_indices = self.compute_indices(entities)
        batch_index_to_entity_index = torch.tensor([i for _, i in sorted(sum([[(bi.item(), ei) for ei, bi in enumerate(ev)] for k, ev in entity_indices.items()], []))], device=self.device)
        num_entities = len(entities[self.schema.id_property.name])
        rev_adjacencies = {k : v.T for k, v in adjacencies.items()}
        property_encodings = self.encode_properties(entities)
        encoded_entities = self.create_autoencoder_inputs(property_encodings, entity_indices)
        bottlenecks, encoded_entities = self.run_first_autoencoder_layer(encoded_entities)
        for depth in range(1, self.depth + 1):
            bottlenecks, encoded_entities = self.run_structured_autoencoder_layer(depth,                
                                                                                  encoded_entities,
                                                                                  prev_bottlenecks=bottlenecks,
                                                                                  adjacencies=adjacencies,
                                                                                  batch_index_to_entity_index=batch_index_to_entity_index,
                                                                                  entity_indices=entity_indices)
        resized_encoded_entities = self.project_autoencoder_outputs(encoded_entities)
        decoded_properties = self.decode_properties(resized_encoded_entities)
        decoded_entities = self.assemble_entities(decoded_properties, entity_indices)
        normalized_entities = {k : self.property_decoders[k].normalize(v.cpu().detach()) for k, v in decoded_entities.items()}
        for k, v in entities.items():
            if k not in decoded_entities:
                decoded_entities[k] = v
                normalized_entities[k] = v
        bottlenecks_by_id = {}
        for entity_type_name, bns in bottlenecks.items():
            #print(entities[self.schema.id_property.name])
            ids = entities[self.schema.id_property.name][entity_indices[entity_type_name].cpu()]
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
    
