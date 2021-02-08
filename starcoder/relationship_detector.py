import re
import argparse
import torch
import json
import numpy
import scipy.sparse
import gzip
from torch.utils.data import DataLoader, Dataset
import logging
from abc import ABCMeta, abstractproperty, abstractmethod

logger = logging.getLogger(__name__)


class RelationshipDetector(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(RelationshipDetector, self).__init__()


class MLP(RelationshipDetector):
    def __init__(self, source_size, target_size):
        super(MLP, self).__init__()
        self.input_size = source_size + target_size
        sizes = [self.input_size, self.input_size // 2, self.input_size // 4]
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
        self.layers = torch.nn.ModuleList(self.layers)
        self.linear = torch.nn.Linear(sizes[-1], 2)
        self.final = torch.nn.Linear(2, 2)
    def forward(self, x):
        for l in self.layers:
            x = torch.nn.functional.relu(l(x))
        return self.final(self.linear(x))
