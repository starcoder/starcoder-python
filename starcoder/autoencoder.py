import torch
import logging
from starcoder.base import StarcoderObject
from starcoder.activation import Activation
from torch import Tensor
from torch.nn import Module
from abc import ABCMeta, abstractproperty, abstractmethod

logger = logging.getLogger(__name__)


class Autoencoder(StarcoderObject, Module, metaclass=ABCMeta):
    def __init__(self, entity_type_name, depth, input_size, bottleneck_size, output_size):
        super(Autoencoder, self).__init__()
        self.entity_type_name = entity_type_name
        self.depth = depth
        self.input_size = input_size
        self.output_size = output_size
        self.bottleneck_size = bottleneck_size


class NullAutoencoder(Autoencoder):
    def __init__(self, entity_type_name, depth):
        super(NullAutoencoder, self).__init__(entity_type_name, depth, 0, 0, 0)
    def forward(self, x):
        logger.debug(
            "Running depth-%d NullAutoencoder for '%s' entities", 
            self.depth, 
            self.entity_type_name
        )
        return (
            torch.zeros(
                size=(x.shape[0], 0),
                device=x.device
            ),
            torch.zeros(
                size=(x.shape[0], 0), 
                device=x.device
            )
        )

class IdentityAutoencoder(Autoencoder):
    def __init__(self, entity_type_name, depth, input_size, bottleneck_size, layer_sizes, activation):
        super(BasicAutoencoder, self).__init__(
            entity_type_name,
            depth,
            input_size,
            bottleneck_size,
            0 if input_size == 0 else layer_sizes[0]
        )
    def forward(self, x):
        return (x, x)


class BasicAutoencoder(Autoencoder):
    def __init__(self, entity_type_name, depth, input_size, bottleneck_size, layer_sizes, activation):
        super(BasicAutoencoder, self).__init__(
            entity_type_name,
            depth,
            input_size,
            bottleneck_size if len(layer_sizes) > 0 else input_size,
            0 if input_size == 0 else input_size if layer_sizes == [] else layer_sizes[0]
        )
        self.encoding_layers = []
        if len(layer_sizes) > 0:
            encoding_layers = [torch.nn.Linear(input_size, layer_sizes[0])]
            decoding_layers = []
            full_layer_sizes = layer_sizes + [bottleneck_size]
            for i in range(len(full_layer_sizes) - 1):
                encoding_layers.append(torch.nn.Linear(full_layer_sizes[i], full_layer_sizes[i + 1]))
                decoding_layers.append(torch.nn.Linear(full_layer_sizes[i + 1], full_layer_sizes[i]))
            encoding_layers = [torch.nn.Identity()] if len(encoding_layers) == 0 or input_size == 0 else encoding_layers
            decoding_layers = [torch.nn.Identity()] if len(decoding_layers) == 0 or input_size == 0 else decoding_layers
            self.encoding_layers = torch.nn.ModuleList(encoding_layers)
            self.decoding_layers = torch.nn.ModuleList(reversed(decoding_layers))
        self.activation = activation
    def forward(self, x):
        if self.encoding_layers == []:
            return (x, x)
        for layer in self.encoding_layers:
            x = self.activation(layer(x))
        bottleneck = x.clone().detach()
        for layer in self.decoding_layers:
            x = self.activation(layer(x))
        #output = self.dropout(x)
        return (x, bottleneck)
