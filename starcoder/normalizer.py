from abc import ABCMeta, abstractmethod, abstractproperty
import torch
from torch import Tensor
import torch.autograd.profiler as profiler
from starcoder.base import StarcoderObject

class Normalizer(StarcoderObject, torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, entity_type_name):
        self.entity_type_name = entity_type_name
        super(Normalizer, self).__init__()

    def forward(self, x:Tensor):
        with profiler.record_function("NORMALIZE {}".format(self.entity_type_name)):
            return self._forward(x)

    @abstractmethod
    def _forward(self, x: Tensor) -> Tensor: pass


class LayerNormalizer(Normalizer):

    def __init__(self, entity_type_name, input_size):
        super(LayerNormalizer, self).__init__(entity_type_name)
        self.norm = torch.nn.LayerNorm([input_size])

    def _forward(self, x):
        return self.norm(x)
