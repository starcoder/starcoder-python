import numpy
import math
import time
import calendar
import torch
import logging
from datetime import date, datetime
from starcoder.base import StarcoderObject
from abc import ABCMeta, abstractmethod, abstractproperty
from torch import Tensor, dtype
import torchvision

logger = logging.getLogger(__name__)


class Property(StarcoderObject, metaclass=ABCMeta):
    """
Property objects represent a type with particular semantics and its canonical
representation.
    """
    def __init__(self, name, spec, **args):
        self.name = name
        self.type_name = spec["type"]
        self.args = args
        self.empty = True
        self.spec = spec
    def __str__(self):
        return "{0}: {1}".format(self.name, self.type_name)
    @abstractproperty
    def missing_value(self): pass


class MetaProperty(Property):
    def __init__(self, name, spec, **args):
        super(MetaProperty, self).__init__(name, spec, **args)
    @property
    def missing_value(self):
        raise Exception("'{}' is a meta-field, so missing values should never occur!".format(self.name))    


class DataProperty(Property): 
    def __init__(self, name, spec, **args):
        super(DataProperty, self).__init__(name, spec, **args)
    def load(self, v): pass
    @abstractmethod
    def pack(self, v): pass
    @abstractmethod
    def unpack(self, v): pass
    @abstractmethod
    def observe_value(self, v): pass
    @abstractproperty
    def stacked_type(self): pass


class EntityTypeProperty(MetaProperty):
    def __init__(self, name: str, **args):
        super(EntityTypeProperty, self).__init__(name, {"type" : "entity_type"}, **args)

    
class RelationshipProperty(Property):
    def __init__(self, name, **args):
        super(RelationshipProperty, self).__init__(name, **args)
        self.source_entity_type = args["source_entity_type"]        
        self.target_entity_type = args["target_entity_type"]
    def __str__(self):
        return "{}({}->{})".format(self.name, self.source_entity_type, self.target_entity_type)
    @property
    def missing_value(self):
        return []


class IdProperty(MetaProperty):
    @property
    def missing_value(self):
        return None
    def __init__(self, name, **args):
        super(IdProperty, self).__init__(name, {"type" : "id"}, **args)

    
class NumericProperty(DataProperty):
    def __init__(self, name, spec, data, **args):
        super(NumericProperty, self).__init__(name, spec, **args)
        self.max_val = None
        self.min_val = None
    def observe_value(self, v):
        if v:
            self.max_val = v if self.max_val == None else max(self.max_val, v)
            self.min_val = v if self.min_val == None else min(self.min_val, v)
            self.empty = False
    def unpack(self, v):
        return v
    def pack(self, v):
        return (self.missing_value if v is None else v)
    @property
    def stacked_type(self):
        return torch.float32    
    @property
    def missing_value(self):
        return float("nan")
    def __str__(self):
        return "{0}: {1}".format(self.name, self.type_name, self.min_val, self.max_val)
    def __len__(self):
        return 1

    
class ScalarProperty(DataProperty):
    def __init__(self, name, spec, data, **args):
        super(ScalarProperty, self).__init__(name, spec, **args)
        self.max_val = None
        self.min_val = None
    def observe_value(self, v):
        if v:
            self.max_val = v if self.max_val == None else max(self.max_val, v)
            self.min_val = v if self.min_val == None else min(self.min_val, v)
            self.empty = False
    def unpack(self, v):
        return v
    def pack(self, v):
        return float(v)
    @property
    def stacked_type(self):
        return torch.float32    
    @property
    def missing_value(self):
        return float("nan")
    def __str__(self):
        return "{0}: {1}".format(self.name, self.type_name, self.min_val, self.max_val)
    def __len__(self):
        return 1


class PlaceProperty(NumericProperty):
    def __len__(self):
        return 2
    def pack(self, v):
        return self.missing_value if v == None else [v["latitude"], v["longitude"]]    
    def unpack(self, v):
        return {"latitude" : v[0].item(), "longitude" : v[1].item()}
    def observe_value(self, v):
        self.empty = False
    @property
    def missing_value(self):
        return [float("nan"), float("nan")]
    @property
    def stacked_type(self):
        return torch.float32
    def __str__(self):
        return "{0}: {1}".format(self.name, self.type_name)    


class DistributionProperty(NumericProperty):
    def __init__(self, name, spec, data, **args):
        labels = set()
        self.is_log = None
        for e in data.entities:
            for k, v in e.get(name, {}).items():
                labels.add(k)
                if v > 0.0:
                    if self.is_log == True:
                        raise Exception("Saw log-probability of {}".format(v))
                    else:
                        self.is_log == False
                if v < 0.0:
                    if self.is_log == False:
                        raise Exception("Saw probability of {}".format(v))
                    else:
                        self.is_log == True
                
        self.label_to_index = {l : i + 1 for i, l in enumerate(labels)}
        self.label_to_index[None] = 0        
        self.index_to_label = {v : k for k, v in self.label_to_index.items()}
        self.eps = 0.000000000000001
        super(DistributionProperty, self).__init__(name, spec, data, **args)
    def __len__(self):
        return len(self.label_to_index)
    def pack(self, v):
        if self.is_log:
            retval = self.missing_value if v == None else [
                float(v.get(self.index_to_label[i], 0.0)) for i in range(len(self.index_to_label))
            ]
        else:
            retval = self.missing_value if v == None else [
                float(math.log(v.get(self.index_to_label[i], 0.0) + self.eps))
                for i in range(len(self.index_to_label))
            ]
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
    @property
    def stacked_type(self):
        return torch.float32
    def __str__(self) -> str:
        return "{0}: {1}".format(self.name, self.type_name)    

    
class DateProperty(NumericProperty):
    def pack(self, v):
        retval = None
        fmts = self.spec["meta"]["format"]
        for fmt in fmts if isinstance(fmts, list) else [fmts]:
            try:
                retval = datetime.strptime(v, fmt).toordinal()
                self.seen_format = fmt
            except:
                pass
        if retval == None:
            raise Exception("Could not parse date string: '{}'".format(v))
        return float(retval)
    def unpack(self, v):
        #try:
        #    for form in self.args["format"] if isinstance(self.args["format"], list) else [self.args["format"]]:
        return None if numpy.isnan(v) or v < 1 else date.fromordinal(int(v)).strftime(self.seen_format)
        #except:
        #    return None


class DateTimeProperty(NumericProperty):
    def pack(self, v):        
        retval = float("nan")
        fmts = self.spec["meta"]["format"]
        for fmt in fmts if isinstance(fmts, list) else [fmts]:
            try:
                retval = datetime.strptime(v, fmt).toordinal()
            except:
                pass
        return retval
    def unpack(self, v):
        try:
            return None if numpy.isnan(v) or v < 1 else datetime.fromordinal(int(v)).strftime(self.args["format"])
        except:
            return None


class ImageProperty(DataProperty):
    def __init__(self, name, spec, data, **args):
        self.width = spec["meta"]["width"]
        self.height = spec["meta"]["height"]
        self.channels = spec["meta"]["channels"]
        self.channel_size = spec["meta"]["channel_size"]
        super(ImageProperty, self).__init__(name, spec, **args)
    def pack(self, v):
        """
        Parameters
        ----------
        v : tensor-like of shape :math: (\text{width}, \text{height}, \text{channels})

        Returns
        -------
        Tensor of shape :math: (\text{channels}, \text{
        """
        retval = self.missing_value if v == None else torch.tensor(v).tolist()
        return retval    
    def unpack(self, v):
        return v if isinstance(v, list) else v.tolist()
    @property
    def stacked_type(self):
        return torch.float32
    @property
    def missing_value(self):
        return numpy.full((self.width, self.height, self.channels), float("nan")).tolist()
    def __len__(self):
        return 1
    def observe_value(self, v):
        if v:
            self.empty = False


class AudioProperty(DataProperty):
    packed_type = torch.float32
    def __init__(self, name, spec, data, **args):
        self.channels = channels
        self.channel_size = channel_size
        super(AudioProperty, self).__init__(name, **args)
    def pack(self, v):
        return numpy.random.random((1000, self.channels)).tolist()
    def unpack(self, v):
        return v
    @property
    def stacked_type(self):
        return torch.float32    
    @property
    def missing_value(self):
        return float("nan")
    def __len__(self):
        return 1
    def observe_value(self, v):
        if v:
            self.empty = False


class VideoProperty(DataProperty):
    packed_type = torch.float32        
    def __init__(self, name, spec, data, **args):
        self.width = width
        self.height = height
        self.channels = channels
        self.channel_size = channel_size
        super(VideoProperty, self).__init__(name, **args)
    def pack(self, v):
        return numpy.random.random((1, self.width, self.height, self.channels)).tolist()
    def unpack(self, v):
        return v
    @property
    def stacked_type(self):
        return torch.float32    
    @property
    def missing_value(self):
        return float("nan")
    def __len__(self):
        return 1
    def observe_value(self, v):
        if v:
            self.empty = False


class CategoricalProperty(DataProperty):
    def __init__(self, name, spec, data, **args):
        super(CategoricalProperty, self).__init__(name, spec, **args)
        self.item_to_id = {}
        self.id_to_item = {}
        for item in data:
            if name in item:
                self.observe_value(item[name])         
    def observe_value(self, v):
        i = self.item_to_id.setdefault(v, len(self.item_to_id) + 1)
        self.id_to_item[i] = v
        self.empty = False
    def pack(self, v):
        # redundant!
        return (self.missing_value if v is None else self.item_to_id.get(v, self.missing_value))
    def unpack(self, v):
        return self.id_to_item.get(v)
    def __str__(self):
        return "{0}: {1}".format(self.name, self.type_name)
    def __len__(self):
        return len(self.item_to_id) + 1
    @property
    def stacked_type(self):
        return torch.int64
    @property
    def missing_value(self):
        return 0


class SequenceProperty(DataProperty):
    def __init__(self, name, spec, data, split_func, join_func, **args):
        super(SequenceProperty, self).__init__(name, spec, **args)
        self.split_func = split_func
        self.join_func = join_func
        self.item_to_id = {}
        self.id_to_item = {}
        self.max_length = 2
        for item in data:
            if name in item:
                self.observe_value(item[name])    
    def observe_value(self, vs):
        values = self.split_func(vs)
        for v in values:
            i = self.item_to_id.setdefault(v, len(self.item_to_id) + 4)
            self.id_to_item[i] = v
        self.max_length = max(len(values) + 2, self.max_length)
        self.empty = False
    def __str__(self):
        return "{0}: {1} ({2} distinct values, max length {3})".format(
            self.name,
            self.type_name,
            len(self),
            self.max_length
        )
    def pack(self, v):
        items = (
            self.missing_value if v is None else [
                self.item_to_id.get(e, self.unknown_value) for e in self.split_func(v)
            ]
        )
        if len(items) > self.max_length:
            items = items[:self.max_length]
        return sum(
            [
                [self.start_value],
                items,
                [self.end_value],
                [self.padding_value] * (self.max_length - len(items))
            ],
            []
        )
    def unpack(self, v):
        return self.join_func([self.id_to_item.get(e, "") for e in v])
    def __len__(self):
        return len(self.id_to_item) + 4              
    @property
    def stacked_type(self):
        return torch.int64
    @property
    def missing_value(self):
        return []
    @property
    def unknown_value(self):
        return 1    
    @property
    def padding_value(self):
        return 0
    @property
    def start_value(self):
        return 2
    @property
    def end_value(self):
        return 3


class CharacterSequenceProperty(SequenceProperty):
    def __init__(self, name, spec, data, **args):
        super(CharacterSequenceProperty, self).__init__(name, spec, data, list, "".join, **args)    


class WordSequenceProperty(SequenceProperty):
    def __init__(self, name, spec, data, **args):
        super(WordSequenceProperty, self).__init__(name, spec, data, str.split, " ".join, **args)


