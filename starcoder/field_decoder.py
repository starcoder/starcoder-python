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
    def __init__(self, *argv: Any, **args) -> None:
        super(FieldDecoder, self).__init__()
        self.encoder = args.get("encoder", None)
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
        
# class NumericDecoder(FieldDecoder):
#     def __init__(self, field: NumericField, input_size: int, activation: Activation, **args: Any) -> None:
#         self.dims = args["dims"]
#         super(NumericDecoder, self).__init__()
#         self._linear = torch.nn.Linear(input_size, self.dims)
#         self.activation = activation
#     def forward(self, x: Tensor) -> Tensor:
#         retval = self.activation(self._linear(x))
#         return retval
#     @property
#     def input_size(self) -> int:
#         return self._linear.in_features
#     @property
#     def output_size(self) -> int:
#         return self._linear.out_features
#     def normalize(self, v):
#         return v
class NumericDecoder(FieldDecoder):
    def __init__(self, field: NumericField, input_size: int, activation: Activation, **args: Any) -> None:
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
        return v.flatten()

class DistributionDecoder(NumericDecoder):
    def __init__(self, field: DistributionField, input_size: int, activation: Activation, **args: Any) -> None:
        args["dims"] = len(field)
        super(DistributionDecoder, self).__init__(field, input_size, activation, **args)


class ScalarDecoder(NumericDecoder):
    def __init__(self, field: NumericField, input_size: int, activation: Activation, **args: Any) -> None:
        super(ScalarDecoder, self).__init__(field, input_size, activation, dims=1)

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
        super(SequenceDecoder, self).__init__(encoder=args.get("encoder", None))
        #print(self.encoder._embeddings.embedding_dim)
        #sys.exit()
        self.field = field
        #print(self.encoder._embeddings(torch.tensor(2)))
        #sys.exit()
        #hs = args.get("hidden_size", 32)
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = 100
        #self.max_length = field.max_length #args.get("rnn_max_decode", 100) #field.max_length)
        self._rnn = rnn_type(self.encoder._embeddings.embedding_dim, input_size, batch_first=True, bidirectional=False)
        #self._rnn = rnn_type(input_size, hs, batch_first=True, bidirectional=False)
        self._classifier = torch.nn.Linear(input_size, len(field))
        self._start_val = torch.zeros(size=(len(self.field),))
        self._start_val[self.field.start_value] = 1.0
        self._start_val = torch.nn.functional.log_softmax(self._start_val, 0)
    def forward(self, x: Tensor) -> Tensor:
        logger.debug("Starting forward pass of TextDecoder for '%s'", self.field.name)
        #previous_value = torch.zeros(size=(x.shape[0],), device=x.device)
        #previous_value[self.field.start_value] = 1
        retval = torch.zeros(size=(x.shape[0], self.max_length, len(self.field)), device=x.device)
        retval[:, 0, :] = self._start_val
        #print(retval[0])
        #print(retval[1])
        #sys.exit()
        x = x.unsqueeze(0)
        #outs = []
        
        #x = torch.stack([x for i in range(self.max_length)], dim=1)
        #inp = torch.stack([torch.empty(size=(x.shape[0], len(self.field))) for i in range(self.max_length)], dim=1)
        #output, _ = self._rnn(x) #inp, torch.unsqueeze(x, 0))
        for i in range(self.max_length - 1):
            iid = self.encoder._embeddings(torch.argmax(retval[:, i, :], 1))
            #print(iid.shape)
            #sys.exit()
            #out, x = self._rnn(retval[:, i, :].unsqueeze(1), x)
            out, x = self._rnn(iid.unsqueeze(1), x)
            cs = self._classifier(out)
            #print(cs.shape)
            cout = torch.log_softmax(cs.squeeze(1), 1) #output[:, i, :])
            
            #print(iid)
            #sys.exit()
            #print(cout.shape, retval.shape)
            retval[:, i + 1, :] = cout
            #outs.append(torch.nn.functional.log_softmax(self._classifier(output[:, i, :]), dim=1))
        #retval = torch.stack(outs, 1)
        #print(retval.shape)
        return retval
    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)
    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)
    def normalize(self, v):
        return numpy.array(v).argmax(2)
