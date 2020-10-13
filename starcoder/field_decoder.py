import logging
from starcoder.base import StarcoderObject
from starcoder.field import DataField, CategoricalField, NumericField, SequenceField, DistributionField
from starcoder.activation import Activation
import torch
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized, cast
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor
import numpy

logger = logging.getLogger(__name__)

class FieldDecoder(StarcoderObject, torch.nn.Module, metaclass=ABCMeta): # type: ignore[type-arg]
    def __init__(self, *argv: Any) -> None:
        super(FieldDecoder, self).__init__()
    def normalize(self, v):
        return v
    
class CategoricalDecoder(FieldDecoder):
    def __init__(self, field: CategoricalField, input_size: int, activation: Activation, **args: Any):
        super(CategoricalDecoder, self).__init__()
        self._input_size = input_size
        self._output_size = len(field)
        self._layerA = torch.nn.Linear(self._input_size, self._input_size * 2)
        self._layerB = torch.nn.Linear(self._input_size * 2, self._output_size)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self._layerA(x))
        x = self._layerB(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x
    @property
    def input_size(self) -> int:
        return self._input_size
    @property
    def output_size(self) -> int:
        return self._output_size
    def normalize(self, v):
        return numpy.array(v).argmax(1)
        
class NumericDecoder(FieldDecoder):
    def __init__(self, field: NumericField, input_size: int, activation: Activation, **args: Any) -> None:
        self.dims = args["dims"]
        super(NumericDecoder, self).__init__()
        self._linear = torch.nn.Linear(input_size, self.dims)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        retval = self.activation(self._linear(x))
        return retval
    @property
    def input_size(self) -> int:
        return self._linear.in_features
    @property
    def output_size(self) -> int:
        return self._linear.out_features
    def normalize(self, v):
        return v

class DistributionDecoder(NumericDecoder):
    def __init__(self, field: DistributionField, input_size: int, activation: Activation, **args: Any) -> None:
        args["dims"] = len(field)
        super(DistributionDecoder, self).__init__(field, input_size, activation, **args)
    
class ScalarDecoder(FieldDecoder):
    def __init__(self, field: NumericField, input_size: int, activation: Activation, **args: Any) -> None:
        super(ScalarDecoder, self).__init__()
        self._linear = torch.nn.Linear(input_size, 1)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        retval = self.activation(self._linear(x))
        return retval
    @property
    def input_size(self) -> int:
        return self._linear.in_features
    @property
    def output_size(self) -> int:
        return self._linear.out_features
    def normalize(self, v):
        return v.flatten()

# class DistributionDecoder(FieldDecoder):
#     def __init__(self, field: DataField, input_size: int, activation: Activation, **args: Any) -> None:
#         super(DistributionDecoder, self).__init__()
#         self._linear = torch.nn.Linear(input_size, len(field.categories))
#         self.activation = activation
#     def forward(self, x: Tensor) -> Tensor:
#         return torch.nn.functional.log_softmax(self.activation(self._linear(x)).squeeze(), dim=1)
#     @property
#     def input_size(self) -> int:
#         return self._linear.in_features
#     @property
#     def output_size(self) -> int:
#         return self._linear.out_features

class AudioDecoder(FieldDecoder):
    def __init__(self, field: DataField, input_size: int, activation: Activation, **args: Any) -> None:
        super(AudioDecoder, self).__init__()                
        self._linear = torch.nn.Linear(input_size, 1)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        retval = self.activation(self._linear(x))
        return retval
    @property
    def input_size(self) -> int:
        return self._linear.in_features
    @property
    def output_size(self) -> int:
        return self._linear.out_features

class VideoDecoder(FieldDecoder):
    def __init__(self, field: DataField, input_size: int, activation: Activation, **args: Any) -> None:
        super(VideoDecoder, self).__init__()                
        self._linear = torch.nn.Linear(input_size, 1)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        retval = self.activation(self._linear(x))
        return retval
    @property
    def input_size(self) -> int:
        return self._linear.in_features
    @property
    def output_size(self) -> int:
        return self._linear.out_features

class ImageDecoder(FieldDecoder):
    def __init__(self, field: DataField, input_size: int, activation: Activation, **args: Any) -> None:
        super(ImageDecoder, self).__init__()                
        self._linear = torch.nn.Linear(input_size, 1)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        retval = self.activation(self._linear(x))
        return retval
    @property
    def input_size(self) -> int:
        return self._linear.in_features
    @property
    def output_size(self) -> int:
        return self._linear.out_features

class SequenceDecoder(FieldDecoder):
    def __init__(self, field: SequenceField, input_size: int, activation: Activation, **args: Any) -> None:
        super(SequenceDecoder, self).__init__()
        self.field = field
        hs = args.get("hidden_size", 32)
        rnn_type = args.get("rnn_type", torch.nn.GRU)        
        self.max_length = args.get("max_length", 10)
        self._rnn = rnn_type(input_size, hs, batch_first=True, bidirectional=False)
        self._classifier = torch.nn.Linear(hs, len(field))
    def forward(self, x: Tensor) -> Tensor:
        logger.debug("Starting forward pass of TextDecoder for '%s'", self.field.name)
        x = torch.stack([x for i in range(self.max_length)], dim=1)
        output, _ = self._rnn(x)
        outs = []
        for i in range(self.max_length):
            outs.append(torch.nn.functional.log_softmax(self._classifier(output[:, i, :]), dim=1))
        retval = torch.stack(outs, 1)
        return retval
    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)
    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)
    def normalize(self, v):
        return numpy.array(v).argmax(2)
