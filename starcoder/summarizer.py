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
#from starcoder.data_fields import NumericField, DistributionField, CategoricalField, SequentialField, IntegerField, DateField
from typing import Type, List, Dict, Set, Any, Callable, Tuple, Union
from starcoder.base import StarcoderObject #Summarizer, Activation, DataField
from abc import ABCMeta, abstractproperty, abstractmethod
from starcoder.activation import Activation

class Summarizer(torch.nn.Module, metaclass=ABCMeta): # type: ignore[type-arg]
    def __init__(self) -> None:
        super(Summarizer, self).__init__()
        pass

# representations -> summary
# (related_entity_count x bottleneck_size) -> (bottleneck_size)
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

    
class SingleSummarizer(Summarizer):
    def __init__(self, input_size: int, activation: Activation) -> None:
        super(SingleSummarizer, self).__init__()
        self._input_size = input_size
        #self.layer = torch.nn.Identity()
    def forward(self, x: Tensor) -> Tensor:
        #if x.shape[0] == 0:
        #    return torch.zeros(shape=(self._input_size,))
        #else:
        return x[0]
    def input_size(self) -> int:
        return self._input_size
    def output_size(self) -> int:
        return self._input_size

