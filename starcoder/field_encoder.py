import logging
from starcoder.base import StarcoderObject
from starcoder.field import DataField, CategoricalField, NumericField, SequenceField, DistributionField
from starcoder.activation import Activation
import torch
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized, cast
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor

logger = logging.getLogger(__name__)

class FieldEncoder(StarcoderObject, torch.nn.Module, metaclass=ABCMeta): # type: ignore[type-arg]
    def __init__(self, *argv: Any) -> None:
        super(FieldEncoder, self).__init__()
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: pass

class CategoricalEncoder(FieldEncoder):
    def __init__(self, field: CategoricalField, activation: Activation, **args: Any) -> None:
        super(CategoricalEncoder, self).__init__()
        self._embeddings = torch.nn.Embedding(num_embeddings=len(field), embedding_dim=args.get("embedding_size", 32))
    def forward(self, x: Tensor) -> Tensor:
        retval = self._embeddings(x)
        return(retval)
    @property
    def input_size(self) -> int:
        return 1
    @property
    def output_size(self) -> int:
        return self._embeddings.embedding_dim


class NumericEncoder(FieldEncoder):
    def __init__(self, field: NumericField, activation: Activation, **args: Any) -> None:
        self.dims = args["dims"]
        super(NumericEncoder, self).__init__()
    def forward(self, x: Tensor) -> Tensor:
        return x
    #retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
    #    return retval
    @property
    def input_size(self) -> int:
        return self.dims
    @property
    def output_size(self) -> int:
        return self.dims

class DistributionEncoder(NumericEncoder):
    def __init__(self, field: DistributionField, activation: Activation, **args: Any) -> None:
        args["dims"] = len(field)
        super(DistributionEncoder, self).__init__(field, activation, **args)
    
class ScalarEncoder(FieldEncoder):
    def __init__(self, field: NumericField, activation: Activation, **args: Any) -> None:
        super(ScalarEncoder, self).__init__()
    def forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval
    @property
    def input_size(self) -> int:
        return 1
    @property
    def output_size(self) -> int:
        return 1
    
# class DistributionEncoder(FieldEncoder):
#     def __init__(self, field: DataField, activation: Activation, **args: Any) -> None:
#         self._size = len(field.categories)
#         super(DistributionEncoder, self).__init__()
#     def forward(self, x: Tensor) -> Tensor:
#         return x
#     @property
#     def input_size(self) -> int:
#         return self._size
#     @property
#     def output_size(self) -> int:
#         return self._size

class AudioEncoder(FieldEncoder):
    def __init__(self, field: DataField, activation: Activation, **args: Any) -> None:
        super(AudioEncoder, self).__init__()        
    def forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval
    @property
    def output_size(self) -> int:
        return 1

class VideoEncoder(FieldEncoder):
    def __init__(self, field: DataField, activation: Activation, **args: Any) -> None:
        super(VideoEncoder, self).__init__()        
    def forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval
    @property
    def output_size(self) -> int:
        return 1

class ImageEncoder(FieldEncoder):
    def __init__(self, field: DataField, activation: Activation, **args: Any) -> None:
        super(ImageEncoder, self).__init__()        
    def forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval
    @property
    def output_size(self) -> int:
        return 1

class SequenceEncoder(FieldEncoder):
    def __init__(self, field: SequenceField, activation: Activation, **args: Any) -> None:
        super(SequenceEncoder, self).__init__()
        self.field = field
        es = args.get("embedding_size", 32)
        hs = args.get("hidden_size", 64)
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = args.get("max_length", 1)
        self._embeddings = torch.nn.Embedding(num_embeddings=len(field), embedding_dim=es)
        self._rnn = rnn_type(es, hs, batch_first=True, bidirectional=False)
    def forward(self, x: Tensor) -> Tensor:
        logger.debug("Starting forward pass of TextEncoder for '%s'", self.field.name)        
        l = (x != 0).sum(1)
        nonempty = l != 0
        x = x[nonempty]
        if x.shape[0] == 0:
            return torch.zeros(size=(nonempty.shape[0], self.output_size), device=x.device)
        l = l[nonempty]
        embs = self._embeddings(x)
        pk = torch.nn.utils.rnn.pack_padded_sequence(embs, l, batch_first=True, enforce_sorted=False)
        output, h = self._rnn(pk)
        h = h.squeeze(0)
        retval = torch.zeros(size=(nonempty.shape[0], h.shape[1]), device=l.device)
        mask = nonempty.unsqueeze(1).expand((nonempty.shape[0], h.shape[1]))
        retval.masked_scatter_(mask, h)
        logger.debug("Finished forward pass for TextEncoder")
        return retval
    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)
    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)
