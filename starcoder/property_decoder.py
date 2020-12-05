import logging
from starcoder.base import StarcoderObject
from starcoder.property import DataProperty, CategoricalProperty, NumericProperty, SequenceProperty, DistributionProperty
from starcoder.activation import Activation
import torch
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized, cast
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor
import numpy

logger = logging.getLogger(__name__)


class PropertyDecoder(StarcoderObject, torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, *argv: Any, **args) -> None:
        super(PropertyDecoder, self).__init__()
        self.encoder = args.get("encoder", None)
    def normalize(self, v):
        return v


class CategoricalDecoder(PropertyDecoder):
    def __init__(self, field: CategoricalProperty, input_size: int, activation: Activation, **args: Any):
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


class NullDecoder(PropertyDecoder):
    def __init__(self, field: NumericProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(NullDecoder, self).__init__()
        #self.dims = args["dims"]
        #self.linear = torch.nn.Linear(input_size, input_size)
        #self.final = torch.nn.Linear(input_size, self.dims)
        #self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor(0.0)
        #retval = self.final(self.activation(self.linear(x)))
        #return retval
    @property
    def input_size(self) -> int:
        return 0
    #return self.linear.in_features
    @property
    def output_size(self) -> int:
        return 0
    #return self.dims
    def normalize(self, v):
        return v

    
class NumericDecoder(PropertyDecoder):
    def __init__(self, field: NumericProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(NumericDecoder, self).__init__()
        self.dims = args["dims"]
        self.linear = torch.nn.Linear(input_size, input_size)
        self.final = torch.nn.Linear(input_size, self.dims)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        retval = self.final(self.activation(self.linear(x)))
        return retval
    @property
    def input_size(self) -> int:
        return self.linear.in_features
    @property
    def output_size(self) -> int:
        return self.dims
    def normalize(self, v):
        return v


class DistributionDecoder(NumericDecoder):
    def __init__(self, field: DistributionProperty, input_size: int, activation: Activation, **args: Any) -> None:
        args["dims"] = len(field)
        super(DistributionDecoder, self).__init__(field, input_size, activation, **args)
    def forward(self, x: Tensor) -> Tensor:
        retval = torch.nn.functional.log_softmax(self.final(self.activation(self.linear(x))), dim=1)
        return retval


class ScalarDecoder(NumericDecoder):
    def __init__(self, field: NumericProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(ScalarDecoder, self).__init__(field, input_size, activation, dims=1)
    def normalize(self, v):
        return v.flatten()


class AudioDecoder(PropertyDecoder):
    def __init__(self, field: DataProperty, input_size: int, activation: Activation, **args: Any) -> None:
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


class VideoDecoder(PropertyDecoder):
    def __init__(self, field: DataProperty, input_size: int, activation: Activation, **args: Any) -> None:
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


class ImageDecoder(PropertyDecoder):
    def __init__(self, field: DataProperty, input_size: int, activation: Activation, **args: Any) -> None:
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


class SequenceDecoder(PropertyDecoder):
    def __init__(self, field: SequenceProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(SequenceDecoder, self).__init__(encoder=args.get("encoder", None))
        self.field = field
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = 100
        self._rnn = rnn_type(self.encoder._embeddings.embedding_dim, input_size, batch_first=True, bidirectional=False)
        self._classifier = torch.nn.Linear(input_size, len(field))
        self._start_val = torch.zeros(size=(len(self.field),))
        self._start_val[self.field.start_value] = 1.0
        self._start_val = torch.nn.functional.log_softmax(self._start_val, 0)
    def forward(self, x: Tensor) -> Tensor:
        logger.debug("Starting forward pass of TextDecoder for '%s'", self.field.name)
        retval = torch.zeros(size=(x.shape[0], self.max_length, len(self.field)), device=x.device)
        if x.shape[0] == 0:
            return retval
        retval[:, 0, :] = self._start_val
        x = x.unsqueeze(0)
        for i in range(self.max_length - 1):
            iid = self.encoder._embeddings(torch.argmax(retval[:, i, :], 1))
            out, x = self._rnn(iid.unsqueeze(1), x)
            cs = self._classifier(out)
            cout = torch.log_softmax(cs.squeeze(1), 1)
            retval[:, i + 1, :] = cout
        return retval
    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)
    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)
    def normalize(self, v):
        return numpy.array(v).argmax(2)
