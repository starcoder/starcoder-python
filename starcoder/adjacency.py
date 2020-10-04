import logging
from scipy.sparse import csr_matrix
import numpy
import torch
from typing import List, Dict, Any, Tuple, Sequence, Hashable, MutableSequence, NewType, cast


logger = logging.getLogger(__name__)

Adjacency = csr_matrix
Adjacencies = Dict[str, csr_matrix]

def stack_adjacencies(to_stack: List[Adjacencies]) -> Adjacencies:
    retval: Dict[str, Dict[str, List[int]]] = {}
    offset = 0
    for adjs in to_stack:
        step: int = 0
        for name, adj in adjs.items():
            step = adj.shape[0]
            retval[name] = retval.get(name, {"values" : [], "rows" : [], "columns" : []})
            for r, c in zip(*adj.nonzero()):
                retval[name]["values"].append(True)
                retval[name]["rows"].append(r + offset)
                retval[name]["columns"].append(c + offset)
        offset += step
    return {k : torch.tensor(csr_matrix((v["values"], (v["rows"], v["columns"])),
                                        shape=(offset, offset),
                                        dtype=numpy.bool).todense()) for k, v in retval.items()}
