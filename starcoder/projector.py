"""A Projector is used to transform Autoencoder outputs for different entity-types, which
are likely of different sizes, to a common size (by default, the maximum of the entity-type
output sizes, to allow lossless propagation).  Their signature is:

  (entity_type_boundary_size) -> (projected_size)

where "projected_size" is constant throughout the model.
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
import torch.nn.functional as F
from starcoder.base import StarcoderObject
from starcoder.activation import Activation
from typing import Type, List, Dict, Set, Any, Optional, Callable, cast
from abc import ABCMeta, abstractmethod, abstractproperty
import torch.autograd.profiler as profiler

class Projector(StarcoderObject, torch.nn.Module, metaclass=ABCMeta): # type: ignore[type-arg]
    def __init__(self, entity_type_name) -> None:
        self.entity_type_name = entity_type_name
        super(Projector, self).__init__()
    def forward(self, x:Tensor):
        with profiler.record_function("PROJECT {}".format(self.entity_type_name)):
            return self._forward(x)
    @abstractmethod
    def _forward(self, x: Tensor) -> Tensor: pass


class MLPProjector(Projector):
    def __init__(self, entity_type, in_size: int, out_size: int, activation: Activation):
        super(MLPProjector, self).__init__(entity_type)
        self.layer = torch.nn.Linear(in_size, out_size)
        self.activation = activation
    def _forward(self, x: Tensor) -> Tensor:
        return self.activation(self.layer(x))
    def input_size(self) -> int:
        return cast(int, self.layer.in_size)
    def output_size(self) -> int:
        return cast(int, self.layer.out_size)
