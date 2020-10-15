import logging
from typing import List, Dict, Type, Tuple
from functools import partial

logger = logging.getLogger(__name__)

from starcoder.field import Field, DataField, RelationshipField, IdField, EntityTypeField, NumericField, CategoricalField, WordSequenceField, CharacterSequenceField, DateField, PlaceField, DateTimeField, DistributionField, ImageField, VideoField, AudioField, ScalarField
field_classes: Dict[str, Type[Field]] = {
    "scalar" : ScalarField,
    "distribution" : DistributionField,
    "place" : PlaceField,
    "date" : DateField,
    "datetime" : DateTimeField,
    "categorical" : CategoricalField,
    "boolean" : CategoricalField,
    "keyword" : CategoricalField,
    "text" : CharacterSequenceField,
    "relationship" : RelationshipField,
    "image" : ImageField,
    "video" : VideoField,
    "audio" : AudioField,
    "id" : IdField,
    "entity_type" : EntityTypeField,
}

from starcoder.field_encoder import FieldEncoder, NumericEncoder, CategoricalEncoder, SequenceEncoder, ScalarEncoder, DistributionEncoder, ImageEncoder, VideoEncoder, AudioEncoder
from starcoder.field_decoder import FieldDecoder, NumericDecoder, CategoricalDecoder, SequenceDecoder, ScalarDecoder, DistributionDecoder, ImageDecoder, VideoDecoder, AudioDecoder
from starcoder.field_loss import FieldLoss, NumericLoss, CategoricalLoss, SequenceLoss, ScalarLoss, ImageLoss, VideoLoss, AudioLoss
field_model_classes: Dict[Type[DataField], Tuple[Type[FieldEncoder], Type[FieldDecoder], Type[FieldLoss]]] = { # type: ignore[type-arg]
    ScalarField : (ScalarEncoder, ScalarDecoder, ScalarLoss),
    CategoricalField : (CategoricalEncoder, CategoricalDecoder, CategoricalLoss),
    CharacterSequenceField : (SequenceEncoder, SequenceDecoder, SequenceLoss),
    WordSequenceField : (SequenceEncoder, SequenceDecoder, SequenceLoss),
    DateField : (ScalarEncoder, ScalarDecoder, ScalarLoss),
    DateTimeField : (ScalarEncoder, ScalarDecoder, ScalarLoss),
    PlaceField : (partial(NumericEncoder, dims=2), partial(NumericDecoder, dims=2), partial(NumericLoss, dims=2)),
    DistributionField : (DistributionEncoder, DistributionDecoder, NumericLoss),
    ImageField : (ImageEncoder, ImageDecoder, ImageLoss),
    VideoField : (VideoEncoder, VideoDecoder, VideoLoss),
    AudioField : (AudioEncoder, AudioDecoder, AudioLoss),    
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


