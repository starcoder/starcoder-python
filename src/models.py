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
import random
import logging
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from data import NumericField, DistributionField, CategoricalField, SequentialField, IntegerField, Missing, Unknown


# (batch_count x (entity_representation_size + (bottleneck_size * relation_count)) :: Float) -> (batch_count x entity_representation_size :: Float)
class Autoencoder(torch.nn.Module):
    def __init__(self, sizes):
        super(Autoencoder, self).__init__()
        self._encoding_layers = []
        self._decoding_layers = []
        for i in range(len(sizes) - 1):
            f = sizes[i] if isinstance(sizes[i], int) else sum(sizes[i])
            t = sizes[i + 1] if isinstance(sizes[i + 1], int) else sum(sizes[i + 1])
            a = sizes[i] if isinstance(sizes[i], int) else sizes[i][0]
            self._encoding_layers.append(torch.nn.Linear(f, t))
            self._decoding_layers.append(torch.nn.Linear(t, a))
        self._encoding_layers = [torch.nn.Identity()] if len(self._encoding_layers) == 0 else self._encoding_layers
        self._decoding_layers = [torch.nn.Identity()] if len(self._decoding_layers) == 0 else self._decoding_layers
        self._encoding_layers = torch.nn.ModuleList(self._encoding_layers)
        self._decoding_layers = torch.nn.ModuleList(reversed(self._decoding_layers))
        self._loss = torch.nn.MSELoss()
        #self._dropout = torch.nn.Dropout(p=0.5)
    def forward(self, x):
        y = x.clone()
        #x = self._dropout(x)
        for layer in self._encoding_layers:
            x = torch.nn.functional.relu(layer(x))
        bottleneck = x.clone().detach()
        for layer in self._decoding_layers:
            x = torch.nn.functional.relu(layer(x))
        return (x, bottleneck, self._loss(x, y))
    @property
    def input_size(self):
        return self._encoding_layers[0].in_features    
    @property
    def output_size(self):
        return self._decoding_layers[-1].out_features        


#
class Projector(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self._layer = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.nn.functional.relu(self._layer(x))


# (batch_size :: Int) -> (batch_size x field_representation_size :: Float)
class CategoricalEncoder(torch.nn.Module):
    def __init__(self, field, **args):
        super(CategoricalEncoder, self).__init__()        
        self._embeddings = torch.nn.Embedding(num_embeddings=len(field), embedding_dim=args.get("embedding_size", 128))

    def forward(self, x):
        return self._embeddings(x)

    @property
    def input_size(self):
        return 1
    @property
    def output_size(self):
        return self._embeddings.embedding_dim


# (batch_size x entity_representation_size :: Float) -> (batch_size x item_types :: Float)
class CategoricalDecoder(torch.nn.Module):
    def __init__(self, field, input_size, **args): #input_size):
        super(CategoricalDecoder, self).__init__()
        output_size = len(field)
        self._layerA = torch.nn.Linear(input_size, output_size)
        self._layerB = torch.nn.Linear(output_size, output_size)
    def forward(self, x):
        x = torch.nn.functional.relu(self._layerA(x))
        x = self._layerB(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x
    @property
    def input_size(self):
        return self._layer.in_features
    @property
    def output_size(self):
        return self._layer.out_features        


CategoricalLoss = torch.nn.NLLLoss


# (batch_count :: Float) -> (batch_count :: Float)
class NumericEncoder(torch.nn.Module):
    def __init__(self, field, **args):
        super(NumericEncoder, self).__init__()
    def forward(self, x):
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval
    @property
    def input_size(self):
        return 1
    @property
    def output_size(self):
        return 1


# (batch_count x entity_representation_size :: Float) -> (batch_count :: Float)    
class NumericDecoder(torch.nn.Module):
    def __init__(self, field, input_size, **args):
        super(NumericDecoder, self).__init__()
        self._linear = torch.nn.Linear(input_size, 1)
    def forward(self, x):
        retval = torch.nn.functional.relu(self._linear(x))
        retval = self._linear(torch.nn.functional.relu(x))
        return retval.squeeze()
    @property
    def input_size(self):
        return self._linear.in_features
    @property
    def output_size(self):
        return self._linear.out_features


NumericLoss = torch.nn.MSELoss


# (batch_count :: Float) -> (batch_count :: Float)
class DistributionEncoder(torch.nn.Module):
    def __init__(self, field, **args):
        self._size = len(field._categories)
        super(DistributionEncoder, self).__init__()
    def forward(self, x):
        return x #torch.unsqueeze(x, 1)
    @property
    def input_size(self):
        return self._size
    @property
    def output_size(self):
        return self._size


# (batch_count x entity_representation_size :: Float) -> (batch_count :: Float)    
class DistributionDecoder(torch.nn.Module):
    def __init__(self, field, input_size, **args):
        super(DistributionDecoder, self).__init__()
        self._linear = torch.nn.Linear(input_size, len(field._categories))
    def forward(self, x):
        return torch.nn.functional.log_softmax(torch.nn.functional.relu(self._linear(x)).squeeze(), dim=1)
    @property
    def input_size(self):
        return self._linear.in_features
    @property
    def output_size(self):
        return self._linear.out_features


DistributionLoss = torch.nn.KLDivLoss


# item_sequences -> lengths -> hidden_state
# (batch_count x max_length :: Int) -> (batch_count :: Int) -> (batch_count x entity_representation_size :: Float)
class SequentialEncoder(torch.nn.Module):
    def __init__(self, field, **args):
        super(SequentialEncoder, self).__init__()
        es = args.get("embedding_size", 32)
        hs = args.get("hidden_size", 64)
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self._embeddings = torch.nn.Embedding(num_embeddings=len(field), embedding_dim=es)
        self._rnn = rnn_type(es, hs, batch_first=True, bidirectional=False)
    def forward(self, x):
        l = (x != Missing.value).sum(1)
        nonempty = l != 0
        x = x[nonempty]
        l = l[nonempty]
        embs = self._embeddings(x)
        pk = torch.nn.utils.rnn.pack_padded_sequence(embs, l, batch_first=True, enforce_sorted=False)
        output, h = self._rnn(pk)
        h = h.squeeze(0)
        retval = torch.zeros(size=(nonempty.shape[0], h.shape[1]), device=l.device)
        mask = nonempty.unsqueeze(1).expand((nonempty.shape[0], h.shape[1]))
        retval.masked_scatter_(mask, h)
        return retval
    @property
    def input_size(self):
        return self._rnn.input_size
    @property
    def output_size(self):
        return self._rnn.hidden_size


# representations -> item_distributions
# (batch_count x entity_representation_size :: Float) -> (batch_count x max_length x item_types :: Float)
class SequentialDecoder(torch.nn.Module):
    def __init__(self, field, input_size, **args): #hidden_size, rnn_type=torch.nn.GRU):
        super(SequentialDecoder, self).__init__()
        hs = args.get("hidden_size", 128)
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self._max_length = field._max_length
        self._rnn = rnn_type(input_size, hs, batch_first=True, bidirectional=False)
        self._classifier = torch.nn.Linear(hs, len(field))
    def forward(self, x):
        x = torch.stack([x for i in range(self._max_length)], dim=1)
        output, _ = self._rnn(x)
        outs = []
        for i in range(self._max_length):
            outs.append(torch.nn.functional.log_softmax(self._classifier(output[:, i, :]), dim=1))
        return torch.stack(outs, 1)
    @property
    def input_size(self):
        return self._rnn.input_size
    @property
    def output_size(self):
        return self._rnn.hidden_size


class SequentialLoss(object):
    def __init__(self, reduction):
        self._nll = torch.nn.NLLLoss(reduction=reduction)
    def __call__(self, x, target):
        x = x[:, 0:target.shape[1], :]
        losses = []
        for v in range(target.shape[1]):
            losses.append(self._nll(x[:, v, :], target[:, v]))
        return torch.cat(losses)


# representations -> summary
# (bottleneck_size x relation_count) -> (bottleneck_size)
class RNNSummarizer(torch.nn.Module):
    def __init__(self, input_size, rnn_type=torch.nn.GRU):
        super(RNNSummarizer, self).__init__()
        self._rnn = rnn_type(input_size, input_size, batch_first=True)
    def forward(self, representations):
        out, h = self._rnn(representations)
        return h


class DummyAutoencoder(torch.nn.Identity):
    def __init__(self):
        super(DummyAutoencoder, self).__init__()
    def forward(self, x):
        return (x, None, None)


class DummyProjector(torch.nn.Identity):
    def __init__(self, size):
        super(DummyProjector, self).__init__()
        self._size = size
    def forward(self, x):
        return torch.zeros(size=(x.shape[0], self._size), device=x.device)


field_models = {
    NumericField : (NumericEncoder, NumericDecoder, NumericLoss(reduction="none")),
    DistributionField : (DistributionEncoder, DistributionDecoder, DistributionLoss(reduction="none")),
    IntegerField : (NumericEncoder, NumericDecoder, NumericLoss(reduction="none")),
    CategoricalField : (CategoricalEncoder, CategoricalDecoder, CategoricalLoss(reduction="none")),
    SequentialField : (SequentialEncoder, SequentialDecoder, SequentialLoss(reduction="none")),
}


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, spec, depth, autoencoder_shapes, embedding_size, hidden_size, mask, field_dropout, hidden_dropout, ae_loss=False, summarizers=RNNSummarizer, projected_size=None):
        super(GraphAutoencoder, self).__init__()

        self._device = torch.device('cpu')
        self._ae_loss = ae_loss
        self._autoencoder_shapes = autoencoder_shapes
        self._mask = mask
        self._field_dropout = field_dropout
        self._hidden_dropout = hidden_dropout
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._bottleneck_size = None if autoencoder_shapes in [[], None] else autoencoder_shapes[-1]

        # entity and field specification
        self._spec = spec

        # order in which to process entity-types
        self._entity_type_order = sorted(spec.entity_types)

        # order in which to process each entity-type's fields
        self._entity_type_field_order = {et : sorted([f for f in spec.entity_fields(et) if type(spec.field_object(f)) in field_models], key=lambda x : str(x)) for et in spec.entity_types}

        # order in which to process each entity-type's relations
        self._entity_type_relation_order = {et : sorted([x for x in spec.entity_relations(et)]) for et in spec.entity_types}

        # how many hops to propagate signal in the graph
        self._depth = depth        

        # encoded widths of entities
        self._boundary_sizes = {}

        # an encoder for each field
        self._field_encoders = {}
        for field_name in self._spec.field_names:
            field_type = type(self._spec.field_object(field_name))
            if field_type in field_models:
                self._field_encoders[field_name] = field_models[field_type][0](self._spec._field_name_to_object[field_name])
        self._field_encoders = torch.nn.ModuleDict(self._field_encoders)

        for entity_type in self._entity_type_order:
            for field_name in self._entity_type_field_order[entity_type]:
                self._boundary_sizes[entity_type] = self._boundary_sizes.get(entity_type, 0)
                self._boundary_sizes[entity_type] += self._field_encoders[field_name].output_size

        #an autoencoder for each entity type and depth
        self._entity_autoencoders = {}        
        for entity_type in self._entity_type_order:
            boundary_size = self._boundary_sizes.get(entity_type, 0)
            if self._bottleneck_size == None or boundary_size == 0:
                self._entity_autoencoders[entity_type] = [DummyAutoencoder()]
            else:
                self._entity_autoencoders[entity_type] = [Autoencoder([boundary_size] + self._autoencoder_shapes)]
                for _ in self._spec.entity_relations(entity_type):
                    boundary_size += self._bottleneck_size
                for depth in range(self._depth):
                    self._entity_autoencoders[entity_type].append(Autoencoder([boundary_size] + self._autoencoder_shapes))

            self._entity_autoencoders[entity_type] = torch.nn.ModuleList(self._entity_autoencoders[entity_type])
        self._entity_autoencoders = torch.nn.ModuleDict(self._entity_autoencoders)

        # a summarizer for each entity-to-entity relation
        if self._depth > 0:
            self._summarizers = {}
            for entity_type in self._entity_type_order:
                self._summarizers[entity_type] = {}
                for name in spec.entity_relations(entity_type):
                    self._summarizers[entity_type][name] = summarizers(self._bottleneck_size)
                self._summarizers[entity_type] = torch.nn.ModuleDict(self._summarizers[entity_type])
            self._summarizers = torch.nn.ModuleDict(self._summarizers)

        # MLP for each entity type to project representations to a common size
        # note the largest boundary size
        self._projected_size = projected_size if projected_size != None else max(self._boundary_sizes.values())
        self._projectors = {}
        for entity_type in self._entity_type_order:
            boundary_size = self._boundary_sizes.get(entity_type, 0)
            if boundary_size == 0:
                self._projectors[entity_type] = DummyProjector(self._projected_size)
            else:
                self._projectors[entity_type] = Projector(self._boundary_sizes.get(entity_type, 0), self._projected_size)
        self._projectors = torch.nn.ModuleDict(self._projectors)
        
        # a decoder for each field
        # change from per-entity-per-field to just per-field!
        # a decoder for each field
        self._field_decoders = {}
        for field_name in self._spec.field_names:
            field_type = type(self._spec.field_object(field_name))
            if field_type in field_models:
                self._field_decoders[field_name] = field_models[field_type][1](self._spec._field_name_to_object[field_name], self._projected_size)
        self._field_decoders = torch.nn.ModuleDict(self._field_decoders)

        # self._field_decoders = {}
        # for entity_type in self._entity_type_order:
        #     boundary_size = self._boundary_sizes[entity_type]
        #     self._field_decoders[entity_type] = {}
        #     for field_name in self._entity_type_field_order[entity_type]:
        #         field_type = type(self._spec.field_object(field_name))
        #         field = self._spec.field_object(field_name)
        #         if field_type in field_models:
        #             self._field_decoders[entity_type][field_name] = field_models[type(field)][1](field, boundary_size)
        #     self._field_decoders[entity_type] = torch.nn.ModuleDict(self._field_decoders[entity_type])
        # self._field_decoders = torch.nn.ModuleDict(self._field_decoders)

    def cuda(self):
        self._device = torch.device('cuda:0')
        super(GraphAutoencoder, self).cuda()
        
    def forward(self, entities, adjacencies):
        logging.debug("Starting forward pass")
        logging.debug("Entities: %s", entities)
        logging.debug("Adjacencies: %s", adjacencies)

        device = self._device
        entity_type_field = self._spec.entity_type_field
        num_entities = entities[entity_type_field].shape[0]
        entity_masks = {}
        autoencoder_inputs = {}        

        for entity_type in self._entity_type_order:
            idx = (entities[entity_type_field] == self._spec.field_object(entity_type_field)[entity_type]).nonzero().flatten()
            if len(idx) > 0:
                entity_masks[entity_type] = idx
        logging.debug("Constructed entity masks: %s", entity_masks)
        
        for entity_type, entity_mask in entity_masks.items():
            autoencoder_inputs[entity_type] = []
            for field_name in self._entity_type_field_order[entity_type]:
                if field_name in entities:
                    field_values = entities[field_name][entity_mask]
                    encodings = self._field_encoders[field_name](field_values)
                else:
                    encodings = torch.zeros(size=(len(entity_mask), #num_entities, 
                                                  self._field_encoders[field_name].output_size),
                                            device=self._device, dtype=torch.float32)
                autoencoder_inputs[entity_type].append(encodings)
            autoencoder_inputs[entity_type].append(torch.zeros(size=(len(entity_mask), 0), dtype=torch.float32, device=self._device))
            autoencoder_inputs[entity_type] = torch.cat(autoencoder_inputs[entity_type], 1)
        logging.debug("Constructed autoencoder inputs: %s", autoencoder_inputs)
            
        autoencoder_outputs = {}
        bottlenecks = torch.zeros(size=(num_entities, self._bottleneck_size), device=self._device) if self._bottleneck_size != None else None
        keep_bottlenecks = self._bottleneck_size != None

        # zero-depth autoencoder
        for entity_type, entity_mask in entity_masks.items():
            entity_outputs, bns, losses = self._entity_autoencoders[entity_type][0](autoencoder_inputs[entity_type])
            autoencoder_outputs[entity_type] = entity_outputs

            #output_mask = entity_masks[entity_name].unsqueeze(1).expand(entity_outputs.shape)
            #bottleneck_mask = entity_masks[entity_name].unsqueeze(1).expand(bottlenecks.shape)
            #autoencoder_outputs.masked_scatter_(output_mask, entity_outputs) #mask, h)            
            if keep_bottlenecks and self._depth > 0:
                bottlenecks[entity_mask] = bns
        logging.debug("Output from zero-depth autoencoder: %s", autoencoder_outputs)

                
        # n-depth autoencoders
        for depth in range(1, self._depth + 1):
            for entity_type, entity_mask in entity_masks.items():
                other_reps = []
                #for rel_name, other_type in self._entity_type_relation_order[entity_type]:
                for rel_name in self._entity_type_relation_order[entity_type]:
                    summarize = self._summarizers[entity_type][rel_name]
                    relation_reps = torch.zeros(size=(len(entity_masks[entity_type]), self._bottleneck_size), device=self._device)
                    for i, idx in enumerate(entity_masks[entity_type]):
                        oidx = (adjacencies[rel_name][idx] == True).nonzero().flatten() if rel_name in adjacencies else []
                        if len(oidx) > 0:
                            neighbors = bottlenecks[oidx].unsqueeze(0)
                            relation_reps[i] = summarize(neighbors).squeeze()
                    other_reps.append(relation_reps)
                sh = list(autoencoder_outputs[entity_type].shape)
                sh[1] = 0
                other_reps = torch.cat(other_reps, 1) if len(other_reps) > 0 else torch.zeros(size=tuple(sh), device=self._device)
                autoencoder_inputs[entity_type] = torch.cat([autoencoder_outputs[entity_type], other_reps], 1)
                entity_outputs, bns, losses = self._entity_autoencoders[entity_type][depth](autoencoder_inputs[entity_type])
                if keep_bottlenecks:
                    bottlenecks[entity_mask] = bns
            logging.debug("Output from %d-depth autoencoder: %s", depth, autoencoder_outputs)
                    
        resized_autoencoder_outputs = {k : self._projectors[k](v) for k, v in autoencoder_outputs.items()}

        # reconstruct the entities by unfolding the last autoencoder output
        reconstructions = {}
        for entity_type, entity_mask in entity_masks.items():
            for field_name in self._entity_type_field_order.get(entity_type, []):
                out = self._field_decoders[field_name](resized_autoencoder_outputs[entity_type])
                if field_name not in reconstructions:
                    reconstructions[field_name] = torch.zeros(size=[num_entities] + list(out.shape)[1:], device=self._device)
                logging.debug("Reconstructing %s: %s", field_name, out)
                reconstructions[field_name][entity_mask] = out
        reconstructions[self._spec.id_field] = entities[self._spec.id_field]
        reconstructions[self._spec.entity_type_field] = entities[self._spec.entity_type_field]
        logging.debug("Reconstructions: %s", reconstructions)
        return (reconstructions, bottlenecks)


    # Recursively initialize model weights
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv1d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
