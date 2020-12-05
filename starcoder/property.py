import numpy
import math
import time
import calendar
import torch
import logging
from datetime import date, datetime
from starcoder.base import StarcoderObject
from typing import Type, List, Dict, Set, Any, Callable, Iterator, Union, Tuple, Sequence, Sized, TypeVar, Generic, Hashable, Optional
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor, dtype

logger = logging.getLogger(__name__)

PackedValueType = Union[int, float]
PackedValue = Union[int, float]
UnpackedValueType = Union[str, int, float]
UnpackedValue = Union[str, int, float]

class Missing():
    pass

I = TypeVar("I")
U = TypeVar("U")
P = TypeVar("P")

class Property(StarcoderObject, metaclass=ABCMeta):
    """
Property objects represent a type with particular semantics and its canonical
representation.
    """
    def __init__(self, name: str, **args: Any) -> None:
        self.name = name
        self.type_name = args["type"]
        self.args = args
        self.empty = True

    def __str__(self) -> str:
        return "{0}: {1}".format(self.name, self.type_name)
    #@abstractproperty
    #def elastic_field(self) -> Dict[str, Any]: pass
    @abstractproperty
    def missing_value(self) -> P: pass

class MetaProperty(Property):
    def __init__(self, name: str, **args: Any) -> None:
        super(MetaProperty, self).__init__(name, **args)
    @property
    def missing_value(self) -> P:
        raise Exception("'{}' is a meta-field, so missing values should never occur!".format(self.name))    
        
class DataProperty(Property, Generic[I, U, P], Sized): 
    def __init__(self, name: str, **args: Any) -> None:
        super(DataProperty, self).__init__(name, **args)
    #@abstractmethod
    def load(self, v: I) -> U: pass
    @abstractmethod
    def pack(self, v: Optional[U]) -> P: pass
    @abstractmethod
    def unpack(self, v: P) -> Optional[U]: pass
    @abstractmethod
    def observe_value(self, v: U) -> None: pass
    #@abstractproperty
    #def unpacked_type(self) -> Any: pass
    #@abstractproperty
    #def packed_type(self) -> Any: pass
    @abstractproperty
    def stacked_type(self) -> dtype: pass
    
class EntityTypeProperty(MetaProperty):
    def __init__(self, name: str, **args: Any) -> None:
        super(EntityTypeProperty, self).__init__(name, type="entity_type", **args)
    @property
    def elastic_field(self) -> Dict[str, Any]:
        return {"type" : "keyword"}
    
class RelationshipProperty(Property):
    def __init__(self, name: str, **args: Any) -> None:
        super(RelationshipProperty, self).__init__(name, **args)
        self.source_entity_type = args["source_entity_type"]        
        self.target_entity_type = args["target_entity_type"]
    def __str__(self) -> str:
        return "{}({}->{})".format(self.name, self.source_entity_type, self.target_entity_type)
    #@property
    #def elastic_field(self) -> Dict[str, Any]:
    #    return {"type" : "join"}
    @property
    def missing_value(self) -> Any:
        return []
    
class IdProperty(MetaProperty):
    @property
    def missing_value(self) -> Any:
        return None
    def __init__(self, name: str, **args: Any) -> None:
        super(IdProperty, self).__init__(name, type="id", **args)
    #@property
    #def elastic_field(self) -> Dict[str, Any]:
    #    return {"type" : "keyword"}

    
class NumericProperty(DataProperty[None, float, float]):
    #packed_type = torch.float32
    #missing_value = float("nan")
    def __init__(self, name: str, **args: Any) -> None:
        super(NumericProperty, self).__init__(name, **args)
        self.max_val = None
        self.min_val = None
    def observe_value(self, v: float) -> None:
        if v:
            self.max_val = v if self.max_val == None else max(self.max_val, v) # type: ignore
            self.min_val = v if self.min_val == None else min(self.min_val, v) # type: ignore
            self.empty = False
        #try:
        #    retval = float(v)
        #    self.max_val = retval if self.max_val == None else max(self.max_val, v) # type: ignore
        #    self.min_val = retval if self.min_val == None else min(self.min_val, v) # type: ignore
        #    self.empty = False
        #    return float(retval)
        #except Exception as e:
        #    logger.error("Could not interpret '%s' for NumericProperty '%s'", v, self.name)
        #    raise e
    def unpack(self, v: List[float]) -> Optional[float]: # type: ignore
        return v
        #if isinstance(v, torch.Tensor):
        #    v = v.item()
        #return (None if numpy.isnan(v) else v)
    def pack(self, v: Optional[float]) -> float:
        return (self.missing_value if v is None else v)
    #@property
    #def packed_type(self) -> dtype:
    #    return torch.float32
    @property
    def stacked_type(self) -> torch.dtype:
        return torch.float32    
    @property
    def missing_value(self) -> float:
        return float("nan")
    def __str__(self) -> str:
        #return "{1} field: {0}[{2}, {3}]".format(self.name, self.type_name, self.min_val, self.max_val)
        return "{0}: {1}".format(self.name, self.type_name, self.min_val, self.max_val)
    def __len__(self) -> int:
        return 1
    #@property
    #def elastic_field(self) -> Dict[str, Any]:
    #    return {"type" : "float"}

    
class ScalarProperty(DataProperty[None, float, float]):
    #packed_type = torch.float32
    #missing_value = float("nan")
    def __init__(self, name: str, **args: Any) -> None:
        super(ScalarProperty, self).__init__(name, **args)
        self.max_val = None
        self.min_val = None
    def observe_value(self, v: float) -> None:
        if v:
            self.max_val = v if self.max_val == None else max(self.max_val, v) # type: ignore
            self.min_val = v if self.min_val == None else min(self.min_val, v) # type: ignore
            self.empty = False
        #try:
        #    retval = float(v)
        #    self.max_val = retval if self.max_val == None else max(self.max_val, v) # type: ignore
        #    self.min_val = retval if self.min_val == None else min(self.min_val, v) # type: ignore
        #    self.empty = False
        #    return float(retval)
        #except Exception as e:
        #    logger.error("Could not interpret '%s' for NumericProperty '%s'", v, self.name)
        #    raise e
    def unpack(self, v: List[float]) -> Optional[float]: # type: ignore
        return v
        #if isinstance(v, torch.Tensor):
        #    v = v.item()
        #return (None if numpy.isnan(v) else v)
    def pack(self, v: Optional[float]) -> float:
        return (self.missing_value if v is None else v)
    #@property
    #def packed_type(self) -> dtype:
    #    return torch.float32
    @property
    def stacked_type(self) -> torch.dtype:
        return torch.float32    
    @property
    def missing_value(self) -> float:
        return float("nan")
    def __str__(self) -> str:
        #return "{1} field: {0}[{2}, {3}]".format(self.name, self.type_name, self.min_val, self.max_val)
        return "{0}: {1}".format(self.name, self.type_name, self.min_val, self.max_val)
    def __len__(self) -> int:
        return 1
    #@property
    #def elastic_field(self) -> Dict[str, Any]:
    #    return {"type" : "float"}


    
class DateProperty(NumericProperty):
    def pack(self, v):
        retval = float("nan")
        for fmt in self.args["format"] if isinstance(self.args["format"], list) else [self.args["format"]]:
            try:
                retval = datetime.strptime(v, fmt).toordinal()
            except:
                pass
        return retval

    def unpack(self, v):
        try:
            for form in self.args["format"] if isinstance(self.args["format"], list) else [self.args["format"]]:
                return None if numpy.isnan(v) or v < 1 else date.fromordinal(int(v)).strftime(form)
        except:
            return None

class DateTimeProperty(NumericProperty):
    def pack(self, v):        
        retval = float("nan")
        for fmt in self.args["format"] if isinstance(self.args["format"], list) else [self.args["format"]]:
            try:
                retval = datetime.strptime(v, fmt).toordinal()
            except:
                pass
        return retval
        #return float("nan") if v == None else datetime.strptime(v, self.args["format"]).toordinal()

    def unpack(self, v):
        try:
            return None if numpy.isnan(v) or v < 1 else datetime.fromordinal(int(v)).strftime(self.args["format"])
        except:
            return None

class ImageProperty(DataProperty[str, List[List[List[int]]], float]):
    #packed_type = torch.float32    
    def __init__(self, name: str, width: int, height: int, channels: int, channel_size: int, **args: Any) -> None:
        self.width = width
        self.height = height
        self.channels = channels
        self.channel_size = channel_size
        super(ImageProperty, self).__init__(name, **args)
    def pack(self, v: Any) -> Any:
        return 1.0
        #return numpy.random.random((self.width, self.height, self.channels)).tolist()
    def unpack(self, v: Any) -> Any:
        return v
    @property
    def stacked_type(self) -> torch.dtype:
        return torch.float32    
    @property
    def missing_value(self) -> float:
        return float("nan") #numpy.full((self.width, self.height, self.channels), float("nan")).tolist()
    def __len__(self) -> int:
        return 1
    def observe_value(self, v: float) -> None:
        if v:
            self.empty = False

class AudioProperty(DataProperty[str, List[int], int]):
    packed_type = torch.float32
    def __init__(self, name: str, channels: int, channel_size: int, **args: Any) -> None:
        #self.data_path = data_path
        self.channels = channels
        self.channel_size = channel_size
        super(AudioProperty, self).__init__(name, **args)
    def pack(self, v: Any) -> Any:
        return numpy.random.random((1000, self.channels)).tolist()
    def unpack(self, v: Any) -> Any:
        return v
    @property
    def stacked_type(self) -> torch.dtype:
        return torch.float32    
    @property
    def missing_value(self) -> float:
        return float("nan") #numpy.full((1, self.channels), float("nan")).tolist()
    def __len__(self) -> int:
        return 1
    def observe_value(self, v: float) -> None:
        if v:
            self.empty = False

class VideoProperty(DataProperty[str, List[List[List[List[int]]]], int]):
    packed_type = torch.float32        
    def __init__(self, name: str, width: int, height: int, channels: int, channel_size: int, **args: Any) -> None:
        #self.data_path = data_path
        self.width = width
        self.height = height
        self.channels = channels
        self.channel_size = channel_size
        super(VideoProperty, self).__init__(name, **args)
    def pack(self, v: Any) -> Any:
        return numpy.random.random((1, self.width, self.height, self.channels)).tolist()
    def unpack(self, v: Any) -> Any:
        return v
    @property
    def stacked_type(self) -> torch.dtype:
        return torch.float32    
    @property
    def missing_value(self) -> float:
        return float("nan") #numpy.full((1000, self.width, self.height, self.channels), float("nan")).tolist()
    def __len__(self) -> int:
        return 1
    def observe_value(self, v: float) -> None:
        if v:
            self.empty = False

class CategoricalProperty(DataProperty[None, str, int]):
    #missing_value = 0
    #packed_type = int
    def __init__(self, name: str, **args: Any) -> None:
        super(CategoricalProperty, self).__init__(name, **args)
        self.item_to_id: Dict[str, int] = {} #Missing() : self.missing_value}
        self.id_to_item: Dict[int, str] = {} #self.missing_value : Missing()}
         
    def observe_value(self, v: str) -> None:
        i = self.item_to_id.setdefault(v, len(self.item_to_id) + 1)
        self.id_to_item[i] = v
        self.empty = False
   
    def pack(self, v: Optional[str]) -> int:
        # redundant!
        return (self.missing_value if v is None else self.item_to_id.get(v, self.missing_value)) #.get(v, self.missing_value)

    def unpack(self, v: List[int]) -> Optional[str]: # type: ignore
        #vv = numpy.array(v)
        #print(v)
        #sys.exit()
        #print(v)
        return self.id_to_item.get(v) #vv.argmax())
        #if isinstance(v, torch.Tensor):
        #    if v.dtype == torch.int64:
        #        v = v.item()
        #    else:
        #        v = v.argmax().item()                
        #if v not in self._rlookup:
        #    raise Exception("Could not unpack value '{0}' (type={1})".format(v, type(v)))
        #return self._rlookup[v]
    
    def __str__(self) -> str:
        #return "{1} field: {0}[{2}]".format(self.name, self.type_name, len(self))
        return "{0}: {1}".format(self.name, self.type_name)
    def __len__(self) -> int:
        return len(self.item_to_id) + 1
    @property
    def stacked_type(self) -> torch.dtype:
        return torch.int64
    @property
    def missing_value(self) -> int:
        return 0

class SequenceProperty(DataProperty[None, Any, List[int]]):
    def __init__(self, name: str, split_func, join_func, **args: Any) -> None:
        super(SequenceProperty, self).__init__(name, **args)
        self.split_func = split_func
        self.join_func = join_func
        self.item_to_id: Dict[str, int] = {} #{None : 0}
        self.id_to_item: Dict[int, str] = {} #{0 : None}
        self.max_length = 2
    
    def observe_value(self, vs: Any) -> None:
        values = self.split_func(vs)
        for v in values:
            i = self.item_to_id.setdefault(v, len(self.item_to_id) + 4)
            self.id_to_item[i] = v
        self.max_length = max(len(values) + 2, self.max_length)
        self.empty = False

    def __str__(self) -> str:
        return "{0}: {1} ({2} distinct values, max length {3})".format(self.name, self.type_name, len(self), self.max_length)

    def pack(self, v: Optional[Any]) -> List[int]:
        #items = (self.missing_value if v is None else [self.item_to_id.get(e, self.unknown_value) for e in self.split_func(v)[0:self.max_length]])
        items = (self.missing_value if v is None else [self.item_to_id.get(e, self.unknown_value) for e in self.split_func(v)])
        #items = ([] if v is None else [self.item_to_id.get(e, self.unknown_value) for e in self.split_func(v)])
        return [self.start_value] + items + [self.end_value] + [self.padding_value] * (self.max_length - len(items))

    def unpack(self, v: Any) -> Optional[Any]:
        try:
            return self.join_func([self.id_to_item.get(e, "") for e in v])
        except:
            return ""

    def __len__(self) -> int:
        return len(self.id_to_item) + 4
              
    @property
    def stacked_type(self) -> torch.dtype:
        return torch.int64
    @property
    def missing_value(self) -> List[int]:
        return []
    @property
    def unknown_value(self) -> int:
        return 1    
    @property
    def padding_value(self) -> int:
        return 0
    @property
    def start_value(self) -> int:
        return 2
    @property
    def end_value(self) -> int:
        return 3
        
class CharacterSequenceProperty(SequenceProperty):
    def __init__(self, name: str, **args: Any):
        super(CharacterSequenceProperty, self).__init__(name, list, "".join, **args)    

class WordSequenceProperty(SequenceProperty):
    def __init__(self, name: str, **args: Any):
        super(WordSequenceProperty, self).__init__(name, str.split, " ".join, **args)

class PlaceProperty(NumericProperty):
    def __len__(self):
        return 2

    def pack(self, v):
        return self.missing_value if v == None else [v["latitude"], v["longitude"]]
    
    def unpack(self, v):
        return {"latitude" : v[0], "longitude" : v[1]}

    def observe_value(self, v):
        self.empty = False

    @property
    def missing_value(self):
        return [float("nan"), float("nan")]

    @property
    def stacked_type(self):
        return torch.float32

    def __str__(self) -> str:
        return "{0}: {1}".format(self.name, self.type_name)    


class DistributionProperty(NumericProperty):
    def __init__(self, name: str, **args: Any):
        self.label_to_index = {}
        self.index_to_label = {}
        super(DistributionProperty, self).__init__(name, **args)

    def __len__(self):
        return len(self.label_to_index)

    def pack(self, v):
        #return self.missing_value if v == None else [v.get(self.index_to_label[i], float("-inf")) for i in range(len(self.index_to_label))]
        #retval = self.missing_value if v == None else [math.log(v.get(self.index_to_label[i], float(0.000000000000001))) for i in range(len(self.index_to_label))]
        if self.is_log:
            retval = self.missing_value if v == None else [v.get(self.index_to_label[i], 0.0) for i in range(len(self.index_to_label))]
        else:
            retval = self.missing_value if v == None else [math.log(v.get(self.index_to_label[i], float(0.000000000000000000000000000000000001))) for i in range(len(self.index_to_label))]
        #print(sum(retval))
        return retval
    
    def unpack(self, v):
        return {self.index_to_label[i] : v[i] for i in range(len(self.index_to_label))}

    def observe_value(self, value):
        self.is_log = None
        self.empty = False
        for k, v in value.items():
            if self.is_log == None:
                if v > 0:
                    self.is_log = False
                elif v < 0:
                    self.is_log = True
            else:
                if v > 0 and self.is_log == True:
                    raise Exception("Saw positive value (%d), but expected log-probabilities", v)
                elif v < 0 and self.is_log == False:
                    raise Exception("Saw negative value (%d), but expected probabilities", v)
            i = self.label_to_index.setdefault(k, len(self.label_to_index))
            self.index_to_label[i] = k

    @property
    def missing_value(self):
        return [0.0 for i in range(len(self.index_to_label))]
        #return [float("nan") for i in range(len(self.index_to_label))]

    @property
    def stacked_type(self):
        return torch.float32

    def __str__(self) -> str:
        return "{0}: {1}".format(self.name, self.type_name)    
    
