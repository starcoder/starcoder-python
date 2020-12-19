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


class Summarizer(torch.nn.Module, metaclass=ABCMeta): # type: ignore[type-arg]
    def __init__(self) -> None:
        super(Summarizer, self).__init__()


class RNNSummarizer(Summarizer):
    def __init__(self, input_size: int, activation: Activation, rnn_type: Callable[..., Callable[[Tensor], Tensor]]=torch.nn.GRU) -> None:
        super(RNNSummarizer, self).__init__()
        self.layer = rnn_type(input_size, input_size, batch_first=True)
    def forward(self, representations: Tensor) -> Tensor:
        out, h = self.layer(representations.unsqueeze(0))
        return h.squeeze()


class MaxPoolSummarizer(Summarizer):
    def __init__(self, input_size: int, activation: Activation) -> None:
        super(MaxPoolSummarizer, self).__init__()
        self.layer = torch.nn.MaxPool1d(1)
    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self.layer(x)

class DANSummarizer(Summarizer):
    def __init__(self, input_size: int, activation: Activation) -> None:
        super(DANSummarizer, self).__init__()
        self.layers = []
        self.sizes = [input_size, input_size, input_size]
        for i in range(len(self.sizes) - 1):
            self.layers.append(torch.nn.Linear(self.sizes[i], self.sizes[i + 1]))
        self.layers = torch.nn.ModuleList(self.layers)
        self.activation = activation
        
    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = torch.mean(x, 0)
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
    
    
class SingleSummarizer(Summarizer):
    def __init__(self, input_size: int, activation: Activation) -> None:
        super(SingleSummarizer, self).__init__()
        self._input_size = input_size
    def forward(self, x: Tensor) -> Tensor:
        return x[0]
    def input_size(self) -> int:
        return self._input_size
    def output_size(self) -> int:
        return self._input_size
