"""A Summarizer creates a fixed-length representation of a variable number of bottlenecks from a single entity-type.

Summarizers are needed because relationships are not one-to-one: for example, a Person 
may have the relationship "owns" with more than one Car, but the Car-bottlenecks need to
occupy a set number of values when added to a graph-aware Person Autoencoder.  Therefore, 
each relationship gets an associated Summarizer with the signature:

  (related_entity_count x related_bottleneck_size) -> (bottleneck_size)
"""
import re
import argparse
import torch
import json
import numpy
import scipy.sparse
import gzip
from torch.utils.data import DataLoader, Dataset
import functools
import numpy
from starcoder.random import random
import logging
from torch import Tensor
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from typing import Type, List, Dict, Set, Any, Callable, Tuple, Union
from starcoder.base import StarcoderObject
from abc import ABCMeta, abstractproperty, abstractmethod
from starcoder.activation import Activation
import torch.autograd.profiler as profiler

logger = logging.getLogger(__name__)


class Summarizer(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, rel_name, position):
        super(Summarizer, self).__init__()
        self.rel_name = rel_name
        self.position = position
    def forward(self, x):
        with profiler.record_function("SUMMARIZER {} {}".format(self.rel_name, self.position)):
            return self._forward(x)
        
    @abstractmethod
    def _forward(self, representations): pass


class NullSummarizer(Summarizer):
    def __init__(self, rel_name, position, input_size):
        super(NullSummarizer, self).__init__(rel_name, position)
    def _forward(self, representations):
        return torch.zeros(size=(representations.shape[0], 0), device=representations.device)
    @property
    def input_size(self):
        return 0
    @property
    def output_size(self):
        return 0

class RNNSummarizer(Summarizer):
    def __init__(self, rel_name, position, input_size, rnn_type=torch.nn.GRU):
        super(RNNSummarizer, self).__init__(rel_name, position)
        self.layer = rnn_type(input_size, input_size, batch_first=True)
    def _forward(self, representations):
        out, h = self.layer(representations.unsqueeze(0))
        return h.squeeze()


class MaxPoolSummarizer(Summarizer):
    def __init__(self, rel_name, position, input_size):
        super(MaxPoolSummarizer, self).__init__(rel_name, position)
        self.layer = torch.nn.MaxPool1d(1)
    def _forward(self, representations):
        return self.layer(representations)


# class DANSummarizer(Summarizer):
#     def __init__(self, rel_name, position, input_size):
#         super(DANSummarizer, self).__init__(rel_name, position)
#         self.layers = []
#         self.sizes = [input_size]
#         for i in range(len(self.sizes) - 1):
#             self.layers.append(torch.nn.Linear(self.sizes[i], self.sizes[i + 1]))
#         self.layers = torch.nn.ModuleList(self.layers)
#         self.activation = torch.nn.functional.relu
#         self._input_size = input_size
#     def _forward(self, representations):
#         x = torch.mean(representations, 0)
#         for layer in self.layers:
#             x = self.activation(layer(x))
#         return x
#     @property
#     def input_size(self):
#         return self._input_size
#     @property
#     def output_size(self):
#         return self._input_size

class DANSummarizer(Summarizer):
    def __init__(self, rel_name, position, input_size):
        super(DANSummarizer, self).__init__(rel_name, position)
        self.layers = []
        self.sizes = [input_size]
        for i in range(len(self.sizes) - 1):
            self.layers.append(torch.nn.Linear(self.sizes[i], self.sizes[i + 1]))
        self.layers = torch.nn.ModuleList(self.layers)
        self.activation = torch.nn.functional.relu
        self._input_size = input_size
    def _forward(self, representations):
        x = torch.mean(representations, 1)
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
    @property
    def input_size(self):
        return self._input_size
    @property
    def output_size(self):
        return self._input_size
    
    
class SingleSummarizer(Summarizer):
    def __init__(self, rel_name, position, input_size):
        super(SingleSummarizer, self).__init__(rel_name, position)
        self._input_size = input_size
    def _forward(self, representations):
        return representations[:, 0, :]
    @property
    def input_size(self):
        return self._input_size
    @property
    def output_size(self):
        return self._input_size
