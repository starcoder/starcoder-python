import re
import argparse
import torch
import json
import numpy
import scipy.sparse
import gzip
#from torch.utils.data import DataLoader, Dataset
import functools
import numpy
from starcoder.random import random
import logging
from torch import Tensor
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from abc import abstractmethod, abstractproperty, ABCMeta
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized
from jsonpath_ng import jsonpath, parse

logger = logging.getLogger(__name__)

#from typing import Type, List, Dict, Set, Any, Union, Tuple

class Configurable(object):
    arguments: List[Any] = []

    def __init__(self, rest: Any=[]) -> None:
        super(Configurable, self).__init__()
        args, rest = self.parse_known_args(rest)
        for k, v in vars(args).items():
            assert re.match(r"^[a-zA-Z_]+$", k) != None
            setattr(self, k, v)    
    
    def parse_known_args(self, rest: Any) -> Any:
        parser = argparse.ArgumentParser()
        for arg in self.arguments: #getattr(cls, "arguments", []):
            parser.add_argument("--{}".format(arg["dest"]), **arg)
        return parser.parse_known_args(rest)

    def get_parse_template(self) -> str:
        template = []
        for arg in getattr(self, "arguments", []) + [{"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : ""}]:
            template.append("--{0} ${{{1}}}".format(arg["dest"], arg["dest"].upper()))
        return " ".join(template)


class Configuration(object):
    # embedding_size = int
    # embedding_init = str
    # autoencoder_shape = [int]
    # max_seq_length = int
    
    def __init__(self, *argv: Any, **argd: Any) -> None:
        self.defaults = {"embedding" : "",
        }
        
    def __getitem__(self, key: Any) -> None:
        pass

    def __setitem__(self, key: Any, value: Any) -> None:
        pass
