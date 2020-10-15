import torch
import logging
from starcoder.base import StarcoderObject
from starcoder.activation import Activation
from torch import Tensor
from torch.nn import Module
from typing import List, Any, Tuple
from abc import ABCMeta, abstractproperty, abstractmethod

logger = logging.getLogger(__name__)

class Autoencoder(StarcoderObject, Module, metaclass=ABCMeta): # type: ignore[type-arg]
    def __init__(self) -> None:
        super(Autoencoder, self).__init__()
    @abstractproperty
    def bottleneck_size(self) -> int: pass
    @abstractproperty
    def input_size(self) -> int: pass
    @abstractproperty
    def output_size(self) -> int: pass    

class BasicAutoencoder(Autoencoder):
    def __init__(self, sizes: List[int], activation: Activation) -> None:
        super(BasicAutoencoder, self).__init__()
        self.sizes = sizes
        encoding_layers: List[Module[Any]] = []
        decoding_layers: List[Module[Any]] = []        
        for i in range(len(self.sizes) - 1):
            encoding_layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            decoding_layers.append(torch.nn.Linear(sizes[i + 1], sizes[i]))
        encoding_layers = [torch.nn.Identity()] if len(encoding_layers) == 0 else encoding_layers
        decoding_layers = [torch.nn.Identity()] if len(decoding_layers) == 0 else decoding_layers
        self.encoding_layers = torch.nn.ModuleList(encoding_layers)
        self.decoding_layers = torch.nn.ModuleList(reversed(decoding_layers))
        #self.loss = torch.nn.MSELoss()
        self.activation = activation
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        y = x.clone()
        y = x
        for layer in self.encoding_layers:
            x = self.activation(layer(x))
        bottleneck = x.clone().detach()
        for layer in self.decoding_layers:
            x = self.activation(layer(x))
        return (x, bottleneck, None) #self.loss(x, y))
    @property
    def input_size(self) -> int:
        return self.sizes[0]
    @property
    def output_size(self) -> int:
        return self.sizes[0]
    @property
    def bottleneck_size(self) -> int:
        return self.sizes[-1]
