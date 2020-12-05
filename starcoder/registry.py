import logging
from typing import List, Dict, Type, Tuple
from functools import partial

logger = logging.getLogger(__name__)

from starcoder.property import Property, DataProperty, RelationshipProperty, IdProperty, EntityTypeProperty, NumericProperty, CategoricalProperty, WordSequenceProperty, CharacterSequenceProperty, DateProperty, PlaceProperty, DateTimeProperty, DistributionProperty, ImageProperty, VideoProperty, AudioProperty, ScalarProperty
property_classes: Dict[str, Type[Property]] = {
    "scalar" : ScalarProperty,
    "distribution" : DistributionProperty,
    "place" : PlaceProperty,
    "date" : DateProperty,
    "datetime" : DateTimeProperty,
    "categorical" : CategoricalProperty,
    "boolean" : CategoricalProperty,
    "keyword" : CategoricalProperty,
    "text" : CharacterSequenceProperty,
    "relationship" : RelationshipProperty,
    "image" : ImageProperty,
    "video" : VideoProperty,
    "audio" : AudioProperty,
    "id" : IdProperty,
    "entity_type" : EntityTypeProperty,
}

from starcoder.property_encoder import PropertyEncoder, NumericEncoder, CategoricalEncoder, SequenceEncoder, ScalarEncoder, DistributionEncoder, ImageEncoder, VideoEncoder, AudioEncoder, DeepAveragingEncoder
from starcoder.property_decoder import PropertyDecoder, NumericDecoder, CategoricalDecoder, SequenceDecoder, ScalarDecoder, DistributionDecoder, ImageDecoder, VideoDecoder, AudioDecoder, NullDecoder
from starcoder.property_loss import PropertyLoss, NumericLoss, CategoricalLoss, SequenceLoss, ScalarLoss, ImageLoss, VideoLoss, AudioLoss, DistributionLoss, NullLoss
property_model_classes: Dict[Type[DataProperty], Tuple[Type[PropertyEncoder], Type[PropertyDecoder], Type[PropertyLoss]]] = { # type: ignore[type-arg]
    ScalarProperty : (ScalarEncoder, ScalarDecoder, ScalarLoss),
    CategoricalProperty : (CategoricalEncoder, CategoricalDecoder, CategoricalLoss),
    CharacterSequenceProperty : (SequenceEncoder, NullDecoder, NullLoss),
    #CharacterSequenceProperty : (SequenceEncoder, SequenceDecoder, SequenceLoss),    
    WordSequenceProperty : (SequenceEncoder, SequenceDecoder, SequenceLoss),
    DateProperty : (ScalarEncoder, ScalarDecoder, ScalarLoss),
    DateTimeProperty : (ScalarEncoder, ScalarDecoder, ScalarLoss),
    PlaceProperty : (partial(NumericEncoder, dims=2), partial(NumericDecoder, dims=2), partial(NumericLoss, dims=2)),
    DistributionProperty : (DistributionEncoder, DistributionDecoder, DistributionLoss),
    ImageProperty : (ImageEncoder, ImageDecoder, ImageLoss),
    VideoProperty : (VideoEncoder, VideoDecoder, VideoLoss),
    AudioProperty : (AudioEncoder, AudioDecoder, AudioLoss),    
}

from starcoder.summarizer import Summarizer, SingleSummarizer, RNNSummarizer, MaxPoolSummarizer
summarizer_classes: Dict[str, Type[Summarizer]] = {
    "single" : SingleSummarizer,
    "rnn" : RNNSummarizer,
    "maxpool" : MaxPoolSummarizer,
}

from starcoder.projector import Projector, MLPProjector
projector_classes: Dict[str, Type[Projector]] = {
    "mlp" : MLPProjector,
}

from starcoder import batchifier
batchifier_classes: Dict[str, Type[batchifier.Batchifier]] = {
    "sample_entities" : batchifier.SampleEntities,
    "sample_snowflakes" : batchifier.SampleSnowflakes,
    "sample_components" : batchifier.SampleComponents,
}

from starcoder import splitter
splitter_classes: Dict[str, Type[splitter.Splitter]] = {
    "sample_entities" : splitter.SampleEntities,
    "sample_components" : splitter.SampleComponents,
}

from starcoder.scheduler import Scheduler, BasicScheduler
scheduler_classes: Dict[str, Type[Scheduler]] = {
    "basic" : BasicScheduler
}


