import logging
from starcoder.base import StarcoderObject
from starcoder.property import DataProperty, CategoricalProperty, NumericProperty, SequenceProperty, DistributionProperty, ScalarProperty
import torch
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized, cast
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor
import torchvision
import torch.autograd.profiler as profiler

logger = logging.getLogger(__name__)


class PropertyEncoder(StarcoderObject, torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, property, *argv: Any) -> None:
        self.property = property
        super(PropertyEncoder, self).__init__()

    def forward(self, x:Tensor):
        with profiler.record_function("ENCODE {}".format(self.property.name)):
            return self._forward(x)

    @abstractmethod
    def _forward(self, x: Tensor) -> Tensor: pass

    @abstractproperty
    def input_size(self): pass


class NullEncoder(PropertyEncoder):

    def __init__(self, property: CategoricalProperty, **args: Any) -> None:
        super(NullEncoder, self).__init__(property)

    def _forward(self, x):
        retval = torch.zeros(size=(x.shape[0], 0), device=x.device)
        return retval

    @property
    def input_size(self):
        return 1

    @property
    def output_size(self):
        return 0


class CategoricalEncoder(PropertyEncoder):

    def __init__(self, property: CategoricalProperty, **args: Any) -> None:
        super(CategoricalEncoder, self).__init__(property)
        self._embeddings = torch.nn.Embedding(num_embeddings=len(property), embedding_dim=args.get("embedding_size", 256))

    def _forward(self, x: Tensor) -> Tensor:
        retval = self._embeddings(x)
        return(retval)

    @property
    def input_size(self) -> int:
        return 1

    @property
    def output_size(self) -> int:
        return self._embeddings.embedding_dim


class NumericEncoder(PropertyEncoder):

    def __init__(self, property: NumericProperty, layers, **args: Any) -> None:
        super(NumericEncoder, self).__init__(property)
        self._layers = torch.nn.ModuleList(layers)
        self.activation = torch.nn.functional.relu
        self.output_size = self._layers[-1].out_features
        self.property = property

    def _forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = self.activation(layer(x))
        return x

    @property
    def layers(self):
        return self._layers

    @property
    def input_size(self) -> int:
        return self._layers[0].in_features


class DistributionEncoder(NumericEncoder):

    def __init__(self, property: DistributionProperty, **args: Any) -> None:        
        super(DistributionEncoder, self).__init__(
            property, 
            torch.nn.ModuleList(
                [
                    torch.nn.Linear(len(property), len(property))
                ]
            ),
            **args
        )


class ScalarEncoder(NumericEncoder):

    def __init__(self, property: ScalarProperty, **args: Any) -> None:
        super(ScalarEncoder, self).__init__(property,
                                            torch.nn.ModuleList(
                                                [
                                                    torch.nn.Linear(1, 10),
                                                    torch.nn.Linear(10, 10)
                                                ]
                                            ),
                                            **args)

    def _forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        for layer in self._layers:
            x = self.activation(layer(x))
        return x


class PlaceEncoder(NumericEncoder):

    def __init__(self, property: ScalarProperty, **args: Any) -> None:
        super(PlaceEncoder, self).__init__(property,
                                            torch.nn.ModuleList(
                                                [
                                                    torch.nn.Linear(2, 10),
                                                    torch.nn.Linear(10, 10)
                                                ]
                                            ),
                                            **args)


class AudioEncoder(PropertyEncoder):

    def __init__(self, property: DataProperty, **args: Any) -> None:
        super(AudioEncoder, self).__init__(property)

    def _forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval

    @property
    def output_size(self) -> int:
        return 1

    @property
    def input_size(self):
        pass


class VideoEncoder(PropertyEncoder):

    def __init__(self, property: DataProperty, **args: Any) -> None:
        super(VideoEncoder, self).__init__(property)

    def _forward(self, x: Tensor) -> Tensor:
        retval = torch.as_tensor(torch.unsqueeze(x, 1), dtype=torch.float32, device=x.device)
        return retval

    @property
    def output_size(self) -> int:
        return 1

    @property
    def input_size(self):
        pass
    
    
class ImageEncoder(PropertyEncoder):
    """Implementation of all-convolutional architecture.
    """

    def __init__(self, property, **args):
        super(ImageEncoder, self).__init__(property)
        reductions = args.get("reductions", 0)
        self.channels = property.channels
        self.output_channels = 1
        layers = [
            torch.nn.Conv2d(self.channels, self.channels, (3,3), stride=1, padding=1),
            torch.nn.Conv2d(self.channels, self.output_channels, (3,3), stride=1, padding=1)
        ]# * 2
        for i in range(reductions):
            pass
        self.property = property
        self.layers = torch.nn.ModuleList(layers)
        self.activation = torch.nn.functional.relu

    def _forward(self, x: Tensor) -> Tensor:
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
        return self.property.width * self.property.height * self.output_channels

    @property
    def input_size(self):
        pass


class GRUEncoder(PropertyEncoder):

    def __init__(self, property: SequenceProperty, **args: Any) -> None:
        super(GRUEncoder, self).__init__(property)
        self.property = property
        es = args.get("embedding_size", 256)
        hs = args.get("hidden_size", 640)
        self._hidden_size = hs
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = property.max_length
        self._embeddings = torch.nn.Embedding(num_embeddings=len(property), embedding_dim=es)
        self._rnn = rnn_type(es, hs, batch_first=True, bidirectional=False, num_layers=2, dropout=0.5)

    def _forward(self, x: Tensor) -> Tensor:
        l = torch.count_nonzero(x, dim=1).cpu()
        x = self._embeddings(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l.cpu().numpy(), batch_first=True, enforce_sorted=False)
        out, ht = self._rnn(x)
        return ht[-1]

    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)

    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)


class AttentionEncoder(PropertyEncoder):

    def __init__(self, property: SequenceProperty, **args: Any) -> None:
        super(AttentionEncoder, self).__init__(property)
        self.property = property
        es = args.get("embedding_size", 128)
        self._embedding_size = es
        self.max_length = property.max_length
        self._embeddings = torch.nn.Embedding(num_embeddings=len(property), embedding_dim=es)
        self._transformer = torch.nn.TransformerEncoderLayer(d_model=es, nhead=8, dropout=0.1)

    def _forward(self, x: Tensor) -> Tensor:
        x = self._embeddings(x)
        out = self._transformer(x)
        out = torch.mean(out, dim=1)
        return out

    @property
    def input_size(self) -> int:
        return None

    @property
    def output_size(self) -> int:
        return self._embedding_size


class CNNEncoder(PropertyEncoder):

    def __init__(self, property: SequenceProperty, **args: Any) -> None:
        super(CNNEncoder, self).__init__(property)
        self.property = property
        es = args.get("embedding_size", 256)
        hs = args.get("hidden_size", 512)
        self._max_width = args.get("width", 4)
        self._cnns = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    es, 
                    es, 
                    i, 
                    padding=self._max_width,
                    padding_mode="zeros") 
                for i in range(1, self._max_width + 1)
            ]
        )
        self._hidden_size = hs
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = property.max_length
        self._embeddings = torch.nn.Embedding(num_embeddings=len(property), embedding_dim=es)
        self._rnn = rnn_type(es * self._max_width, hs, batch_first=True, bidirectional=False, num_layers=2, dropout=0.5)

    def _forward(self, x: Tensor) -> Tensor:
        l = torch.count_nonzero(x, dim=1).cpu()
        x = self._embeddings(x)
        convs = []
        x = x.permute((0, 2, 1))
        for i, net in enumerate(self._cnns, 1):
            val = net(x).permute(0, 2, 1)
            convs.append(val[:, self._max_width - i:x.shape[2] + (self._max_width - i), :])
        x = torch.cat(convs, 2)
        seq_len = x.shape[1]
        batches = x.shape[0]
        hs = x.shape[2]
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l.cpu().numpy(), batch_first=True, enforce_sorted=False)
        out, ht = self._rnn(x)
        return ht[-1]

    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)

    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)


class SequenceEncoder(PropertyEncoder):

    def __init__(self, property: SequenceProperty, **args: Any) -> None:
        super(SequenceEncoder, self).__init__(property)
        self.property = property
        es = args.get("embedding_size", 32)
        hs = args.get("hidden_size", 64)
        self._hidden_size = hs
        rnn_type = args.get("rnn_type", torch.nn.GRU)
        self.max_length = property.max_length
        self._embeddings = torch.nn.Embedding(num_embeddings=len(property), embedding_dim=es)
        self._rnn = rnn_type(es, hs, batch_first=True, bidirectional=False)

    def _forward(self, x: Tensor) -> Tensor:
        logger.debug("Starting forward pass of SequenceEncoder for '%s'", self.property.name)        

        # if no non-empty sequences, return zeros (which won't get gradients)
        if x.shape[0] == 0:
            return torch.zeros(size=(x.shape[0], self.output_size), device=x.device)        

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
            # the pack function requires the lengths to be on CPU
            subset_l = (subset != 0).sum(1).cpu()

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

    def __init__(self, property, **args):
        super(DeepAveragingEncoder, self).__init__(property)
        self.property = property
        self._embeddings = torch.nn.Embedding(num_embeddings=len(property), embedding_dim=es)
        self._rnn = rnn_type(es, hs, batch_first=True, bidirectional=False)

    def _forward(self, x: Tensor) -> Tensor:
        logger.debug("Starting forward pass of SequenceEncoder for '%s'", self.property.name)        

    @property
    def input_size(self) -> int:
        return cast(int, self._rnn.input_size)

    @property
    def output_size(self) -> int:
        return cast(int, self._rnn.hidden_size)
