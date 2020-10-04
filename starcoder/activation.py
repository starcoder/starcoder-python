import logging
from torch import Tensor
from typing import Callable

logger = logging.getLogger(__name__)

Activation = Callable[[Tensor], Tensor]
