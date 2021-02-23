"""A Projector is used to transform Autoencoder outputs for different entity-types, which
are likely of different sizes, to a common size (by default, the maximum of the entity-type
output sizes, to allow lossless propagation).  Their signature is:

  (entity_type_boundary_size) -> (projected_size)

where "projected_size" is constant throughout the model.
"""
import torch
import logging
from torch import Tensor
from starcoder.base import StarcoderObject
from typing import Type, List, Dict, Set, Any, Optional, Callable, cast
from abc import ABCMeta, abstractmethod, abstractproperty
import torch.autograd.profiler as profiler

logger = logging.getLogger(__name__)


class Projector(StarcoderObject, torch.nn.Module, metaclass=ABCMeta):
    
    def __init__(self, entity_type_name) -> None:
        self.entity_type_name = entity_type_name
        super(Projector, self).__init__()

    def forward(self, x:Tensor):
        with profiler.record_function("PROJECT {}".format(self.entity_type_name)):
            return self._forward(x)

    @abstractmethod
    def _forward(self, x: Tensor) -> Tensor: pass


class NullProjector(Projector):

    def __init__(self, entity_type_name, in_size, out_size, activation):
        super(NullProjector, self).__init__(entity_type_name)
        self.out_size = out_size

    def _forward(self, x: Tensor) -> Tensor:
        return torch.zeros(size=(x.shape[0], self.output_size), device=x.device)

    def input_size(self) -> int:
        return None

    @property
    def output_size(self) -> int:
        return self.out_size


class IdentityProjector(Projector):

    def __init__(self, entity_type_name, in_size, out_size, activation):
        super(IdentityProjector, self).__init__(entity_type_name)
        self.out_size = out_size

    def _forward(self, x: Tensor) -> Tensor:
        return x

    def input_size(self) -> int:
        return None

    def output_size(self) -> int:
        return self.out_size


class MLPProjector(Projector):

    def __init__(self, entity_type, in_size: int, out_size: int, activation):
        super(MLPProjector, self).__init__(entity_type)
        self.layer = torch.nn.Linear(in_size, out_size)
        self.activation = activation

    def _forward(self, x: Tensor) -> Tensor:
        return self.activation(self.layer(x))

    def input_size(self) -> int:
        return cast(int, self.layer.in_size)

    def output_size(self) -> int:
        return cast(int, self.layer.out_size)
