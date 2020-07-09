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
from starcoder.fields import NumericField, DistributionField, CategoricalField, SequentialField, IntegerField, DateField #, WordField

logger = logging.getLogger(__name__)

# TODO: add checks for e.g. categoricals with no observed categories

class StarcoderArchitecture(torch.nn.Module):
    def __init__(self):
        super(StarcoderArchitecture, self).__init__()

class Loss(torch.nn.Module):
    def __init__(self, field):
        super(Loss, self).__init__()
        self.field = field
    def __call__(self, guess, gold):
        retval = self.compute(guess, gold)
        assert retval.device == guess.device, self.field.name
        return retval
    def compute(self, guess, gold):
        raise UnimplementedException()

# (batch_count x (entity_representation_size + (bottleneck_size * relation_count)) :: Float) -> (batch_count x entity_representation_size :: Float)
class Autoencoder(torch.nn.Module):
    def __init__(self, sizes, activation):
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
        self._activation = activation()
    def forward(self, x):
        y = x.clone()
        for layer in self._encoding_layers:
            x = self._activation(layer(x))
        bottleneck = x.clone().detach()
        for layer in self._decoding_layers:
            x = self._activation(layer(x))
        return (x, bottleneck, self._loss(x, y))
    @property
    def input_size(self):
        return self._encoding_layers[0].in_features    
    @property
    def output_size(self):
        return self._decoding_layers[-1].out_features        


class DummyAutoencoder(torch.nn.Identity):
    def __init__(self, sizes):
        super(DummyAutoencoder, self).__init__()
        self.input_size = 0
        self.output_size = 0
        self.bottleneck_size = sizes[-1]
    def forward(self, x):
        return (x, torch.zeros(size=(x.shape[0], self.bottleneck_size)), None)

    
#
class Projector(torch.nn.Module):

    def __init__(self, input_size, output_size, activation):
        super(Projector, self).__init__()
        self._layer = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.nn.functional.leaky_relu(self._layer(x))


# (batch_size :: Int) -> (batch_size x field_representation_size :: Float)
class CategoricalEncoder(torch.nn.Module):
    def __init__(self, field, activation, **args):
        super(CategoricalEncoder, self).__init__()
        self._embeddings = torch.nn.Embedding(num_embeddings=len(field), embedding_dim=args.get("embedding_size", 32))
    def forward(self, x):
        retval = self._embeddings(x)
        return(retval)
    @property
    def input_size(self):
        return 1
    @property
    def output_size(self):
        return self._embeddings.embedding_dim


# (batch_size x entity_representation_size :: Float) -> (batch_size x item_types :: Float)
class CategoricalDecoder(torch.nn.Module):
    def __init__(self, field, input_size, activation, **args): #input_size):
        super(CategoricalDecoder, self).__init__()
        output_size = len(field)
        self._layerA = torch.nn.Linear(input_size, input_size * 2)
        self._layerB = torch.nn.Linear(input_size * 2, output_size)
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self._layerA(x))
        x = self._layerB(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x
    @property
    def input_size(self):
        return self._layer.in_features
    @property
    def output_size(self):
        return self._layer.out_features        


#CategoricalLoss = torch.nn.NLLLoss
class CategoricalLoss(Loss):
    def __init__(self, field, reduction="mean"):
        super(CategoricalLoss, self).__init__(field)
        self.reduction = reduction
    def compute(self, guess, gold):
        return torch.nn.functional.cross_entropy(guess, gold)



# (batch_count :: Float) -> (batch_count :: Float)
class NumericEncoder(torch.nn.Module):
    def __init__(self, field, activation, **args):
        super(NumericEncoder, self).__init__()
    def forward(self, x):
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        #retval = torch.as_tensor(x, dtype=torch.float32, device=x.device)
        return retval
    @property
    def input_size(self):
        return 1
    @property
    def output_size(self):
        return 1


# (batch_count x entity_representation_size :: Float) -> (batch_count :: Float)    
class NumericDecoder(torch.nn.Module):
    def __init__(self, field, input_size, activation, **args):
        super(NumericDecoder, self).__init__()
        self._linear = torch.nn.Linear(input_size, 1)
        self.activation = activation()
    def forward(self, x):
        retval = self.activation(self._linear(x))
        return retval
    @property
    def input_size(self):
        return self._linear.in_features
    @property
    def output_size(self):
        return self._linear.out_features


class NumericLoss(Loss):
    def __init__(self, field, reduction="mean"):
        super(NumericLoss, self).__init__(field)
        self.reduction = reduction
    def compute(self, guess, gold):
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold).to(device=guess.device)
        return torch.nn.functional.mse_loss(torch.masked_select(guess, selector), torch.masked_select(gold, selector), reduction=self.reduction)



# (batch_count :: Float) -> (batch_count :: Float)
class DistributionEncoder(torch.nn.Module):
    def __init__(self, field, activation, **args):
        self._size = len(field.categories)
        super(DistributionEncoder, self).__init__()
    def forward(self, x):
        return x
    @property
    def input_size(self):
        return self._size
    @property
    def output_size(self):
        return self._size


# (batch_count x entity_representation_size :: Float) -> (batch_count :: Float)    
class DistributionDecoder(torch.nn.Module):
    def __init__(self, field, input_size, activation, **args):
        super(DistributionDecoder, self).__init__()
        self._linear = torch.nn.Linear(input_size, len(field.categories))
        self.activation = activation()
    def forward(self, x):
        return torch.nn.functional.log_softmax(self.activation(self._linear(x)).squeeze(), dim=1)
    @property
    def input_size(self):
        return self._linear.in_features
    @property
    def output_size(self):
        return self._linear.out_features


#DistributionLoss = torch.nn.KLDivLoss
class DistributionLoss(Loss):
    def __init__(self, field):
        super(DistributionLoss, self).__init__(field)
    def compute(self, guess, gold):
        return torch.nn.functional.kl_div(guess, gold)

# item_sequences -> lengths -> hidden_state
# (batch_count x max_length :: Int) -> (batch_count :: Int) -> (batch_count x entity_representation_size :: Float)
class SequentialEncoder(torch.nn.Module):
    def __init__(self, field, activation, **args):
        super(SequentialEncoder, self).__init__()
        self.field = field
        es = args.get("embedding_size", 32)
        hs = args.get("hidden_size", 64)
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = args.get("max_length", 10)
        self._embeddings = torch.nn.Embedding(num_embeddings=len(field), embedding_dim=es)
        self._rnn = rnn_type(es, hs, batch_first=True, bidirectional=False)
    def forward(self, x):
        logger.debug("Starting forward pass of SequentialEncoder for '%s'", self.field.name)        
        l = (x != 0).sum(1)
        nonempty = l != 0
        x = x[nonempty]
        if x.shape[0] == 0:
            return torch.zeros(size=(nonempty.shape[0], self.output_size))
        l = l[nonempty]
        embs = self._embeddings(x)
        pk = torch.nn.utils.rnn.pack_padded_sequence(embs, l, batch_first=True, enforce_sorted=False)
        output, h = self._rnn(pk)
        h = h.squeeze(0)
        retval = torch.zeros(size=(nonempty.shape[0], h.shape[1]), device=l.device)
        mask = nonempty.unsqueeze(1).expand((nonempty.shape[0], h.shape[1]))
        retval.masked_scatter_(mask, h)
        logger.debug("Finished forward pass for SequentialEncoder")
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
    def __init__(self, field, input_size, activation, **args): #hidden_size, rnn_type=torch.nn.GRU):
        super(SequentialDecoder, self).__init__()
        self.field = field
        hs = args.get("hidden_size", 32)
        rnn_type = args.get("rnn_type", torch.nn.GRU)        
        self.max_length = args.get("max_length", 10)
        self._rnn = rnn_type(input_size, hs, batch_first=True, bidirectional=False)
        self._classifier = torch.nn.Linear(hs, len(field))
    def forward(self, x):
        logger.debug("Starting forward pass of SequentialDecoder for '%s'", self.field.name)
        x = torch.stack([x for i in range(self.max_length)], dim=1)
        output, _ = self._rnn(x)
        outs = []
        for i in range(self.max_length):
            outs.append(torch.nn.functional.log_softmax(self._classifier(output[:, i, :]), dim=1))
        retval = torch.stack(outs, 1)
        return retval
    @property
    def input_size(self):
        return self._rnn.input_size
    @property
    def output_size(self):
        return self._rnn.hidden_size


class SequentialLoss(Loss):
    def __init__(self, field, reduction="mean"):
        super(SequentialLoss, self).__init__(field)
    def compute(self, x, target):
        if target.shape[1] == 0:
            target = torch.zeros(size=x.shape[:-1], device=x.device, dtype=torch.long)
        losses = []
        for v in range(min(x.shape[1], target.shape[1])):
            losses.append(torch.nn.functional.nll_loss(x[:, v, :], target[:, v]))
        return sum(losses)


# representations -> summary
# (related_entity_count x bottleneck_size) -> (bottleneck_size)
class RNNSummarizer(torch.nn.Module):
    def __init__(self, input_size, activation, rnn_type=torch.nn.GRU):
        super(RNNSummarizer, self).__init__()
        self._rnn = rnn_type(input_size, input_size, batch_first=True)
    def forward(self, representations):
        out, h = self._rnn(representations.unsqueeze(0))
        return h.squeeze()


class MaxPoolSummarizer(torch.nn.MaxPool1d):
    def __init__(self, input_size, activation):
        super(MaxPoolSummarizer, self).__init__(1)
    def forward(self, x):
        return super(MaxPoolSummarizer, self).forward(x)

    
class SingleSummarizer(torch.nn.Identity):
    def __init__(self, input_size, activation):
        self._input_size = input_size
        super(SingleSummarizer, self).__init__()
    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros(shape=(self._input_size,))
        else:
            return x[0]


class MLPProjector(torch.nn.Module):
    def __init__(self, in_size, out_size, activation):
        super(MLPProjector, self).__init__()
        self._layer = torch.nn.Linear(in_size, out_size)
    def forward(self, x):
        return self._layer(x)
