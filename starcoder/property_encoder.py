import logging
from starcoder.base import StarcoderObject
from starcoder.property import DataProperty, CategoricalProperty, NumericProperty, SequenceProperty, DistributionProperty, ScalarProperty
from starcoder.activation import Activation
import torch
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized, cast
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor
import torchvision
#import torchaudio

logger = logging.getLogger(__name__)

class PropertyEncoder(StarcoderObject, torch.nn.Module, metaclass=ABCMeta): # type: ignore[type-arg]
    def __init__(self, *argv: Any) -> None:
        super(PropertyEncoder, self).__init__()
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: pass

class NullEncoder(PropertyEncoder):
    def __init__(self, field: CategoricalProperty, activation: Activation, **args: Any) -> None:
        super(NullEncoder, self).__init__()
    def forward(self, x):
        retval = torch.zeros(size=(x.shape[0], 0), device=x.device)
        return retval
    @property
    def input_size(self):
        return 1
    @property
    def output_size(self):
        return 0
    
class CategoricalEncoder(PropertyEncoder):
    def __init__(self, property: CategoricalProperty, activation: Activation, **args: Any) -> None:
        super(CategoricalEncoder, self).__init__()
        self._embeddings = torch.nn.Embedding(num_embeddings=len(property), embedding_dim=args.get("embedding_size", 32))
    def forward(self, x: Tensor) -> Tensor:
        retval = self._embeddings(x)
        return(retval)
    @property
    def input_size(self) -> int:
        return 1
    @property
    def output_size(self) -> int:
        return self._embeddings.embedding_dim

class NumericEncoder(PropertyEncoder):
    def __init__(self, field: NumericProperty, activation: Activation, **args: Any) -> None:
        super(NumericEncoder, self).__init__()
        self.dims = args["dims"]
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(1, 10),
                torch.nn.Linear(10, 10)
            ]
        )
        self.activation = activation
        
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.layers[1](self.activation(self.layers[0](x))))
    @property
    def input_size(self) -> int:
        return self.dims
    @property
    def output_size(self) -> int:
        return self.dims

class DistributionEncoder(PropertyEncoder):
    def __init__(self, field: DistributionProperty, activation: Activation, **args: Any) -> None:        
        self.dims = len(field)
        super(DistributionEncoder, self).__init__(field, activation, **args)
        self.activation = activation

        self._output_size = 128
        self._layerA = torch.nn.Linear(self.dims, self._output_size)
        #self.linearB = torch.nn.Linear(self.dims, self.dims // 2)
    @property
    def input_size(self):
        return self.dims
    @property
    def output_size(self):
        return self._output_size
    def forward(self, x):
        #print(x.sum(1))
        retval = self.activation(self._layerA(x))
        return retval

class ScalarEncoder(NumericEncoder):
    def __init__(self, field: ScalarProperty, activation: Activation, **args: Any) -> None:
        args["dims"] = 1
        super(ScalarEncoder, self).__init__(field, activation, **args)
    def forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval

class AudioEncoder(PropertyEncoder):
    def __init__(self, field: DataProperty, activation: Activation, **args: Any) -> None:
        super(AudioEncoder, self).__init__()        
    def forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval
    @property
    def output_size(self) -> int:
        return 1

class VideoEncoder(PropertyEncoder):
    def __init__(self, field: DataProperty, activation: Activation, **args: Any) -> None:
        super(VideoEncoder, self).__init__()        
    def forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval
    @property
    def output_size(self) -> int:
        return 1

class _ImageEncoder(PropertyEncoder):
    """Implementation of all-convolutional architecture.
    """
    def __init__(self, property: DataProperty, activation: Activation, **args: Any) -> None:
        super(ImageEncoder, self).__init__()
        self.model = torchvision.models.densenet161(pretrained=True)
        for name, p in self.model.named_parameters():
            p.requires_grad = False
        self.model.features = self.model.features[:12]
        #reductions = args.get("reductions", 0)
        #layers = [torch.nn.Conv2d(3, 3, (3,3), stride=1, padding=1)]
        #for i in range(reductions):
        #    pass
        #self.property = property
        #self.layers = torch.nn.ModuleList(layers)
        #self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : tensor of shape :math: (\text{batch}, \text{width}, \text{height}, \text{channels})

        Returns
        -------
        Tensor of shape :math: (\text{batch}, \frac{\text{width} \times \text{height} \times \text{channels}}{2^\text{reductions}})
        """
        x = x.permute((0, 3, 2, 1))
        for i, m in enumerate(self.model.features):
            x = self.model.features[i](x)
        x = x.mean(3).squeeze()
        return x
        index_space = torch.arange(0, x.shape[0], 1, device=x.device, dtype=torch.int64)
        not_nan_mask = ~x.reshape(x.shape[0], -1).sum(1).isnan()
        not_nan = index_space.masked_select(not_nan_mask)
        x = x.index_select(0, not_nan)
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        x = x.reshape(x.shape[0], self.output_size)
        retval = torch.zeros(size=(not_nan_mask.shape[0], self.output_size), device=x.device)
        retval[not_nan] = x
        return retval
    @property
    def output_size(self) -> int:
        return 2208
    
    
class ImageEncoder(PropertyEncoder):
    """Implementation of all-convolutional architecture.
    """
    def __init__(self, property: DataProperty, activation: Activation, **args: Any) -> None:
        super(ImageEncoder, self).__init__()
        reductions = args.get("reductions", 0)
        layers = [torch.nn.Conv2d(3, 3, (3,3), stride=1, padding=1)]
        for i in range(reductions):
            pass
        self.property = property
        self.layers = torch.nn.ModuleList(layers)
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : tensor of shape :math: (\text{batch}, \text{width}, \text{height}, \text{channels})

        Returns
        -------
        Tensor of shape :math: (\text{batch}, \frac{\text{width} \times \text{height} \times \text{channels}}{2^\text{reductions}})
        """
        x = x.permute((0, 3, 2, 1))
        index_space = torch.arange(0, x.shape[0], 1, device=x.device, dtype=torch.int64)
        not_nan_mask = ~x.reshape(x.shape[0], -1).sum(1).isnan()
        not_nan = index_space.masked_select(not_nan_mask)
        x = x.index_select(0, not_nan)
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        x = x.reshape(x.shape[0], self.output_size)
        retval = torch.zeros(size=(not_nan_mask.shape[0], self.output_size), device=x.device)
        retval[not_nan] = x
        return retval
    @property
    def output_size(self) -> int:
        return self.property.width * self.property.height * self.property.channels

    
class SequenceEncoder(PropertyEncoder):
    def __init__(self, field: SequenceProperty, activation: Activation, **args: Any) -> None:
        super(SequenceEncoder, self).__init__()
        self.field = field
        es = args.get("embedding_size", 32)
        hs = args.get("hidden_size", 64)
        self._hidden_size = hs
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = field.max_length #args.get("max_length", 1)
        self._embeddings = torch.nn.Embedding(num_embeddings=len(field), embedding_dim=es)
        self._rnn = rnn_type(es, hs, batch_first=True, bidirectional=False)
    def forward(self, x: Tensor) -> Tensor:
        logger.debug("Starting forward pass of SequenceEncoder for '%s'", self.field.name)        

        # compute lengths of each sequence
        l = (x != 0).sum(1)

        # truncate at the length of longest sequence
        x = x[:, 0:l.max()]

        # determine which sequences are non-empty
        nonempty = l != 0

        # keep the non-empty sequences and lengths
        x = x[nonempty]

        # if no non-empty sequences, return zeros (which won't get gradients)
        if x.shape[0] == 0:
            return torch.zeros(size=(nonempty.shape[0], self.output_size), device=x.device)

        # keep the non-empty sequence lengths
        l = l[nonempty]

        # create tensor to hold outputs
        retval = torch.zeros(size=(nonempty.shape[0], self._hidden_size), device=l.device)

        # create index tensor over batch
        index_space = torch.arange(0, x.shape[0], 1, device=x.device)

        # compute buckets
        bucket_count = min(20, l.shape[0])
        bucket_size = int(x.shape[1] / float(bucket_count))
        buckets = torch.bucketize(l, torch.tensor([i * bucket_size for i in range(bucket_count)], device=x.device))

        # iterate over buckets
        for bucket in buckets.unique():

            # create bucket mask over batch
            bucket_mask = buckets == bucket

            # determine longest sequence in bucket
            subset_max_len = bucket * bucket_size

            # compute bucket indices into batch
            bucket_indices = index_space.masked_select(bucket_mask)

            # select bucket items and truncate to maximum length
            subset = x.index_select(0, bucket_indices)[:, 0:subset_max_len]

            # compute bucket item lengths (should just select from l...)
            subset_l = (subset != 0).sum(1)

            # embed bucket
            subset_embs = self._embeddings(subset)

            if subset_embs.shape[1] > 0:
                # pack bucket
                subset_pk = torch.nn.utils.rnn.pack_padded_sequence(subset_embs, subset_l, batch_first=True, enforce_sorted=False)

                # run RNN
                subset_output, subset_h = self._rnn(subset_pk)

                # get hidden state
                subset_h = subset_h.squeeze(0)

                # assign hidden states to return value
                retval[bucket_indices] = subset_h
        logger.debug("Finished forward pass for SequenceEncoder")
        return retval

    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)
    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)

class DeepAveragingEncoder(SequenceEncoder):
    def __init__(self, property, activation, **args):
        super(DeepAveragingEncoder, self).__init__()
        self.property = property
        #es = args.get("embedding_size", 32)
        #hs = args.get("hidden_size", 64)
        #self._hidden_size = hs
        #rnn_type = args.get("rnn_type", torch.nn.GRU)
        #self.max_length = field.max_length #args.get("max_length", 1)
        self._embeddings = torch.nn.Embedding(num_embeddings=len(field), embedding_dim=es)
        self._rnn = rnn_type(es, hs, batch_first=True, bidirectional=False)
    def forward(self, x: Tensor) -> Tensor:
        logger.debug("Starting forward pass of SequenceEncoder for '%s'", self.field.name)        
    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)
    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)

    
