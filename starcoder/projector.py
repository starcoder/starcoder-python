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

class Projector(StarcoderObject, torch.nn.Module, metaclass=ABCMeta): # type: ignore[type-arg]
    def __init__(self) -> None:
        super(Projector, self).__init__()


class MLPProjector(Projector):
    def __init__(self, in_size: int, out_size: int, activation: Activation):
        super(MLPProjector, self).__init__()
        self.layer = torch.nn.Linear(in_size, out_size)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.layer(x))
    def input_size(self) -> int:
        return cast(int, self.layer.in_size)
    def output_size(self) -> int:
        return cast(int, self.layer.out_size)
