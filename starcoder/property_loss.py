from starcoder.base import StarcoderObject
from starcoder.property import DataProperty, CategoricalProperty, NumericProperty, SequenceProperty, DistributionProperty
import torch
import logging
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized, cast, TypeVar, Hashable, Generic
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor

Guess = TypeVar("Guess")
Gold = TypeVar("Gold")


logger = logging.getLogger(__name__)

class PropertyLoss(StarcoderObject, Generic[Guess, Gold], metaclass=ABCMeta):
    def __init__(self, field: DataProperty[Any, Any, Any]) -> None:
        super(PropertyLoss, self).__init__()
        self.field = field
    @abstractmethod
    def __call__(self, guess: Guess, gold: Gold) -> Tensor: pass
    #retval = self.compute(guess, gold)
    #    assert retval.device == guess.device, self.field.name
    #    return retval
    #@abstractmethod
    #def compute(self, guess: Any, gold: Any) -> Any: pass

class NullLoss(PropertyLoss):
    def __init__(self, property) -> None:
        super(NullLoss, self).__init__(property)
        #self.field = field
    #@method
    def __call__(self, guess: Guess, gold: Gold) -> Tensor:
        return torch.tensor(0.0)
    
    
#CategoricalLoss = torch.nn.NLLLoss
class CategoricalLoss(PropertyLoss[Tensor, Tensor]):
    def __init__(self, field: CategoricalProperty, reduction: str="none") -> None:
        super(CategoricalLoss, self).__init__(field)
        self.reduction = reduction
        self.name = field.name
    def __call__(self, guess: Tensor, gold: Tensor) -> Tensor:
        #selector = gold != 0 #
        selector = torch.nonzero(gold).flatten().to(device=guess.device) #~torch.isnan(gold).to(device=guess.device)
        guess = torch.index_select(guess, 0, selector)
        gold = torch.index_select(gold, 0, selector)
        retval = torch.nn.functional.cross_entropy(guess, gold, reduction=self.reduction)
        return retval



# (batch_count :: Float) -> (batch_count :: Float)


# (batch_count x entity_representation_size :: Float) -> (batch_count :: Float)    


class NumericLoss(PropertyLoss[Tensor, Tensor]):
    def __init__(self, field: NumericProperty, reduction: str="mean", **args) -> None:
        #self.dims = args["dims"]
        super(NumericLoss, self).__init__(field)
        self.reduction = reduction
    def __call__(self, guess: Tensor, gold: Tensor) -> Tensor:
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold) #.to(device=guess.device)
        retval = torch.nn.functional.mse_loss(torch.masked_select(guess, selector), torch.masked_select(gold, selector), reduction=self.reduction)
        return retval

class DistributionLoss(PropertyLoss[Tensor, Tensor]):
    def __init__(self, field: DistributionProperty, reduction: str="mean", **args) -> None:
        #self.dims = args["dims"]
        super(DistributionLoss, self).__init__(field)
        self.reduction = reduction
    def __call__(self, guess: Tensor, gold: Tensor) -> Tensor:
        #print(guess)
        #print(gold)
        #sys.exit()
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold) #.to(device=guess.device)
        #return torch.Tensor([0.0])
        #print(selector)
        #sys.exit()
        retval = torch.nn.functional.kl_div(torch.masked_select(guess, selector), torch.masked_select(gold, selector), reduction=self.reduction, log_target=True)
        #retval = torch.nn.functional.mse_loss(torch.masked_select(guess, selector), torch.masked_select(gold, selector), reduction=self.reduction)
        return retval

class ScalarLoss(NumericLoss):
    def __init__(self, field: NumericProperty, reduction: str="none", **args) -> None:
        super(ScalarLoss, self).__init__(field, reduction, dims=1, **args)

class SScalarLoss(PropertyLoss[Tensor, Tensor]):
    def __init__(self, field: NumericProperty, reduction: str="none") -> None:
        super(ScalarLoss, self).__init__(field)
        self.reduction = reduction
    def __call__(self, guess: Tensor, gold: Tensor) -> Tensor:
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold) #.to(device=guess.device)
        retval = torch.nn.functional.mse_loss(torch.masked_select(guess, selector), torch.masked_select(gold, selector), reduction=self.reduction)
        return retval



# (batch_count :: Float) -> (batch_count :: Float)


# (batch_count x entity_representation_size :: Float) -> (batch_count :: Float)    

#class DistributionLoss(PropertyLoss):
#    def __init__(self, field: DataProperty) -> None:
#        super(DistributionLoss, self).__init__(field)
#    def compute(self, guess: Tensor, gold: Tensor) -> Tensor:
#        return torch.nn.functional.kl_div(guess, gold)



class AudioLoss(PropertyLoss):
    def __init__(self, field: DataProperty, reduction: str="none") -> None:
        super(AudioLoss, self).__init__(field)
        self.reduction = reduction
    def __call__(self, guess: Tensor, gold: Tensor) -> Tensor:
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold)
        retval = torch.nn.functional.mse_loss(torch.masked_select(guess, selector), torch.masked_select(gold, selector), reduction=self.reduction)
        return retval





class VideoLoss(PropertyLoss):
    def __init__(self, field: DataProperty, reduction: str="none") -> None:
        super(VideoLoss, self).__init__(field)
        self.reduction = reduction
    def __call__(self, guess: Tensor, gold: Tensor) -> Tensor:
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold)
        retval = torch.nn.functional.mse_loss(torch.masked_select(guess, selector), torch.masked_select(gold, selector), reduction=self.reduction)
        return retval




class ImageLoss(PropertyLoss):
    def __init__(self, field: DataProperty, reduction: str="none") -> None:
        super(ImageLoss, self).__init__(field)
        self.reduction = reduction
    def __call__(self, guess: Tensor, gold: Tensor) -> Tensor:
        guess = guess.flatten()
        gold = gold.flatten()
        selector = ~torch.isnan(gold)
        retval = torch.nn.functional.mse_loss(torch.masked_select(guess, selector), torch.masked_select(gold, selector), reduction=self.reduction)
        return retval
    

# item_sequences -> lengths -> hidden_state
# (batch_count x max_length :: Int) -> (batch_count :: Int) -> (batch_count x entity_representation_size :: Float)


# representations -> item_distributions
# (batch_count x entity_representation_size :: Float) -> (batch_count x max_length x item_types :: Float)


class SequenceLoss(PropertyLoss[Tensor, Tensor]):
    def __init__(self, field: SequenceProperty, reduction: str="none") -> None:
        super(SequenceLoss, self).__init__(field)
        self.reduction = reduction
    def __call__(self, x: Tensor, target: Tensor) -> Tensor:
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
        return torch.nn.functional.nll_loss(x, target, reduction=self.reduction)
