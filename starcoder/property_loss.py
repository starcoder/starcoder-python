from starcoder.base import StarcoderObject
from starcoder.property import DataProperty, CategoricalProperty, NumericProperty, SequenceProperty, DistributionProperty
import torch
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor
import torchvision
import numpy
import torch.autograd.profiler as profiler

logger = logging.getLogger(__name__)


class PropertyLoss(StarcoderObject, torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, property):
        super(PropertyLoss, self).__init__()
        self.property = property

    def __call__(self, guess, gold):
        with profiler.record_function("LOSS {}".format(self.property.name)):
            return self._call(guess, gold)

    @abstractmethod
    def _call(self, guess, gold): pass

    def _normalize(self, v):
        return v
        
    def normalize(self, v):
        with profiler.record_function("NORMALIZE {}".format(self.property.name)):
            return self._normalize(v)


class NullLoss(PropertyLoss):

    def __init__(self, property):
        super(NullLoss, self).__init__(property)

    def _call(self, guess, gold):
        return torch.tensor(0.0, device=guess.device)


class CategoricalLoss(PropertyLoss):

    def __init__(self, property, reduction="none"):
        super(CategoricalLoss, self).__init__(property)
        self.reduction = reduction

    def _call(self, guess, gold):
        selector = torch.nonzero(gold).flatten()
        guess = torch.index_select(guess, 0, selector)
        gold = torch.index_select(gold, 0, selector)
        retval = torch.nn.functional.cross_entropy(guess, gold, reduction=self.reduction)
        return retval

    def _normalize(self, v):
        return int(numpy.array(v).argmax(0).tolist())


class NumericLoss(PropertyLoss):

    def __init__(self, property, reduction="mean", **args):
        super(NumericLoss, self).__init__(property)
        self.reduction = reduction

    def _call(self, guess, gold):
        guess_ = guess.flatten()
        gold_ = gold.flatten()
        selector = ~torch.isnan(guess_)
        guess_ = torch.masked_select(guess_, selector)
        gold_ = torch.masked_select(gold_, selector)
        retval = torch.nn.functional.mse_loss(guess_, gold_, reduction=self.reduction)
        return retval


class DistributionLoss(PropertyLoss):
    def __init__(self, property, reduction="mean", **args):
        super(DistributionLoss, self).__init__(property)
        self.reduction = reduction
    def _call(self, guess, gold):
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold)
        retval = torch.nn.functional.kl_div(
            torch.masked_select(guess, selector), 
            torch.masked_select(gold, selector), 
            reduction=self.reduction, 
            log_target=True
        )
        return retval


class ScalarLoss(NumericLoss):
    def __init__(self, property, reduction="none", **args):
        super(ScalarLoss, self).__init__(property, reduction, dims=1, **args)
    def _normalize(self, v):
        return v.item()


class AudioLoss(PropertyLoss):
    def __init__(self, property, reduction="none"):
        super(AudioLoss, self).__init__(property)
        self.reduction = reduction
    def _call(self, guess, gold):
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold)
        retval = torch.nn.functional.mse_loss(
            torch.masked_select(guess, selector), 
            torch.masked_select(gold, selector), 
            reduction=self.reduction
        )
        return retval


class VideoLoss(PropertyLoss):
    def __init__(self, property, reduction="none"):
        super(VideoLoss, self).__init__(property)
        self.reduction = reduction
    def _call(self, guess, gold):
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold)
        retval = torch.nn.functional.mse_loss(
            torch.masked_select(guess, selector),
            torch.masked_select(gold, selector),
            reduction=self.reduction
        )
        return retval


class ImageLoss(PropertyLoss):
    def __init__(self, property, reduction="mean"):
        super(ImageLoss, self).__init__(property)
        self.reduction = reduction
    def _call(self, guess, gold):
        guess = guess.flatten()
        gold = gold.flatten()        
        selector = ~torch.isnan(gold)
        guess = torch.masked_select(guess, selector)
        gold = torch.masked_select(gold, selector)
        retval = torch.nn.functional.mse_loss(
            guess, 
            gold, 
            reduction=self.reduction
        )
        return retval
    

class SequenceLoss(PropertyLoss):
    def __init__(self, property, reduction="none"):
        super(SequenceLoss, self).__init__(property)
        self.reduction = reduction
    def _call(self, x, target):
        min_len = min(x.shape[1], target.shape[1])
        if target.shape[1] == 0:
            target = torch.zeros(size=x.shape[:-1], device=x.device, dtype=torch.long)
        target = target[:, 0:min_len]
        x = x[:, 0:min_len, :]
        target = target.flatten(start_dim=0, end_dim=1)
        x = x.flatten(start_dim=0, end_dim=1)
        mask = target != 0
        x = x[mask]
        target = target[mask]
        return torch.nn.functional.cross_entropy(x, target, reduction=self.reduction)
    def _normalize(self, v):
        return [int(x) for x in numpy.array(v).argmax(1).tolist()]
