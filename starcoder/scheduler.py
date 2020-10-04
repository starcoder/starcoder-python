import logging
import warnings
import torch
#from starcoder.base import Scheduler
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Optional, Tuple
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

class Scheduler(object):
    @abstractmethod
    def step(self, metric: float) -> Tuple[bool, bool, bool]: pass


class BasicScheduler(Scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, early_stop: int, *argv: Any, **argdict: Any) -> None:
        self.early_stop = early_stop
        self.num_bad_epochs_for_early_stop = 0
        self.last_epoch = 0
        self.cooldown_counter: int
        self.patience: int        
        super(Scheduler, self).__init__(*argv, **argdict)
        
    def step(self, metrics: float) -> Tuple[bool, bool, bool]: # type: ignore
        is_reduce_rate = False
        is_early_stop = False
        is_new_best = False
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        #if epoch is None:
        epoch = self.last_epoch + 1
        #else:
        #    warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best): # type: ignore
            self.best = current
            is_new_best = True
            self.num_bad_epochs = 0
            self.num_bad_epochs_for_early_stop = 0
        else:
            self.num_bad_epochs += 1
            self.num_bad_epochs_for_early_stop += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
            self.num_bad_epochs_for_early_stop = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch) # type: ignore
            self.cooldown_counter = self.cooldown # type: ignore
            self.num_bad_epochs = 0
            is_reduce_rate = True

        if self.num_bad_epochs_for_early_stop > self.early_stop:
            is_early_stop = True

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups] # type: ignore
        return (is_reduce_rate, is_early_stop, is_new_best)
