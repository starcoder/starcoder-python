import logging
from starcoder.base import StarcoderObject
from starcoder.property import DataProperty, CategoricalProperty, NumericProperty, SequenceProperty, DistributionProperty
from starcoder.activation import Activation
import torch
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized, cast
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor
import numpy
import torchvision
#import torchaudio
import torch.autograd.profiler as profiler

logger = logging.getLogger(__name__)


class PropertyDecoder(StarcoderObject, torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, property, *argv: Any, **args) -> None:
        super(PropertyDecoder, self).__init__()
        self.encoder = args.get("encoder", None)
        self.property = property
    def normalize(self, v):
        return v
    def forward(self, x:Tensor):
        with profiler.record_function("DECODE {}".format(self.property.name)):
            return self._forward(x)
    @abstractmethod
    def _forward(self, x: Tensor) -> Tensor: pass


class CategoricalDecoder(PropertyDecoder):
    def __init__(self, property: CategoricalProperty, input_size: int, activation: Activation, **args: Any):
        super(CategoricalDecoder, self).__init__(property)
        self._input_size = input_size
        self._output_size = len(property)
        self._layerA = torch.nn.Linear(self._input_size, self._output_size)
        self._layerB = torch.nn.Linear(self._output_size, self._output_size)
        self.activation = activation
        self.name = property.name
    def _forward(self, x: Tensor) -> Tensor:
        x = self.activation(self._layerA(x))
        x = self._layerB(x)
        #x = torch.nn.functional.log_softmax(x, dim=1)
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
    def __init__(self, property: NumericProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(NullDecoder, self).__init__(property)
        #self.dims = args["dims"]
        #self.linear = torch.nn.Linear(input_size, input_size)
        #self.final = torch.nn.Linear(input_size, self.dims)
        #self.activation = activation
    def _forward(self, x: Tensor) -> Tensor:
        return torch.zeros(size=(x.shape[0], 0), device=x.device)
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
    def __init__(self, property: NumericProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(NumericDecoder, self).__init__(property)
        self.dims = args["dims"]
        self.linear = torch.nn.Linear(input_size, input_size // 2)
        self.final = torch.nn.Linear(input_size // 2, self.dims)
        self.activation = activation
    def _forward(self, x: Tensor) -> Tensor:
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
    def __init__(self, property: DistributionProperty, input_size: int, activation: Activation, **args: Any) -> None:
        args["dims"] = len(property)
        super(DistributionDecoder, self).__init__(property, input_size, activation, **args)
    def _forward(self, x: Tensor) -> Tensor:
        retval = torch.nn.functional.log_softmax(self.final(self.activation(self.linear(x))), dim=1)
        return retval


class ScalarDecoder(NumericDecoder):
    def __init__(self, property: NumericProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(ScalarDecoder, self).__init__(property, input_size, activation, dims=1)
    def normalize(self, v):
        return v.flatten()

class PlaceDecoder(NumericDecoder):
    def __init__(self, property: NumericProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(PlaceDecoder, self).__init__(property, input_size, activation, dims=2)
    def normalize(self, v):
        return v.flatten()


class AudioDecoder(PropertyDecoder):
    def __init__(self, property: DataProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(AudioDecoder, self).__init__(property)
        self._linear = torch.nn.Linear(input_size, 1)
        self.activation = activation
    def _forward(self, x: Tensor) -> Tensor:
        retval = self.activation(self._linear(x))
        return retval
    @property
    def input_size(self) -> int:
        return self._linear.in_features
    @property
    def output_size(self) -> int:
        return self._linear.out_features


class VideoDecoder(PropertyDecoder):
    def __init__(self, property: DataProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(VideoDecoder, self).__init__(property)
        self._linear = torch.nn.Linear(input_size, 1)
        self.activation = activation
    def _forward(self, x: Tensor) -> Tensor:
        retval = self.activation(self._linear(x))
        return retval
    @property
    def input_size(self) -> int:
        return self._linear.in_features
    @property
    def output_size(self) -> int:
        return self._linear.out_features


class ImageDecoder(PropertyDecoder):
    def __init__(self, property: DataProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(ImageDecoder, self).__init__(property)
        self.channels = 3
        out_size = property.width*property.height*self.channels
        linear_layers = [torch.nn.Linear(input_size, out_size)]

        self.deconvA = torch.nn.Conv2d(self.channels, 3, (3,3), stride=1, padding=1)
        self.deconvB = torch.nn.Conv2d(self.channels, 3, (3,3), stride=1, padding=1)
        self.property = property
        #for _ in range(2):
        #    linear_layers.append(torch.nn.Linear(linear_layers[-1].out_features, out_size, bias=False))
        self.layers = torch.nn.ModuleList(linear_layers)
        self.activation = activation
    def _forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape :math: (\text{batches}, \text{input\_size})

        Returns
        -------
        Tensor of shape :math: (\text{batches}, \text{width}, \text{height}, \text{channels})
        """
        # batch, chan, height, width
        x = self.activation(self.layers[0](x))
        x = x.reshape((x.shape[0], self.channels, self.property.height, self.property.width))
        x = self.activation(self.deconvA(x))
        x = self.deconvB(x)
        x = x.permute(0, 3, 2, 1)
        #return x
        #for i in range(len(self.layers) - 1):
        #    x = self.activation(self.layers[i](x))
        #x = self.layers[-1](x)
        #retval = x.reshape((x.shape[0], self.property.width, self.property.height, self.property.channels))
        return x
    @property
    def input_size(self) -> int:
        return self.layers[0].in_features
    @property
    def output_size(self) -> int:
        return self.layers[-1].out_features


class SequenceDecoder(PropertyDecoder):
    def __init__(self, property: SequenceProperty, input_size: int, activation: Activation, **args: Any) -> None:
        super(SequenceDecoder, self).__init__(property)
        self.property = property
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = 100
        self.hidden_size = 64
        self.input_size = input_size
        self._projector = torch.nn.Linear(input_size, self.hidden_size)
        if "encoder" in args:
            self._embeddings = args["encoder"]._embeddings
        #self._rnn = rnn_type(self.encoder._embeddings.embedding_dim, hidden_size, batch_first=True, bidirectional=False)
        self._rnn_cell = torch.nn.GRUCell(self._embeddings.embedding_dim, self.hidden_size)
        self._classifier = torch.nn.Linear(self.hidden_size, len(property))
        self._start_val = torch.zeros(size=(len(self.property),))
        self._start_val[self.property.start_value] = 1.0
        self._start_val = torch.nn.functional.log_softmax(self._start_val, 0)
    def _forward(self, x: Tensor) -> Tensor:
        #print(x.shape)
        x = self._projector(x)
        logger.debug("Starting forward pass of TextDecoder for '%s'", self.property.name)
        retval = torch.zeros(size=(x.shape[0], self.max_length, len(self.property)), device=x.device)
        if x.shape[0] == 0:
            return retval
        retval[:, 0, :] = self._start_val
        x = x.unsqueeze(0)
        for i in range(self.max_length - 1):
            prev = torch.argmax(retval[:, i, :], 1)
            #if prev == self.property.end_value:
            #    break
            iid = self._embeddings(prev).detach()
            #print(iid.shape, x.shape)
            h = self._rnn_cell(iid, x.squeeze(0))
            #print(h)
            cs = self._classifier(h)
            cout = cs.squeeze(1) #torch.log_softmax(cs.squeeze(1), 1)
            retval[:, i + 1, :] = cout
            #out, x = self._rnn(iid.unsqueeze(1), x)
            #cs = self._classifier(out)
            #cout = torch.log_softmax(cs.squeeze(1), 1)
            #retval[:, i + 1, :] = cout
        return retval
    #@property
    #def input_size(self) -> int:
    #    return self._projector.input_size
    @property
    def output_size(self) -> int:
        return cast(int, self._classifier.output_size)
    def normalize(self, v):
        return numpy.array(v).argmax(2)
