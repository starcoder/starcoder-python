import logging
from typing import List, Dict, Type, Tuple

logger = logging.getLogger(__name__)

from starcoder.field import Field, DataField, RelationshipField, IdField, EntityTypeField, NumericField, CategoricalField, WordSequenceField, CharacterSequenceField
field_classes: Dict[str, Type[Field]] = {
    "numeric" : NumericField,
    "categorical" : CategoricalField,
    "boolean" : CategoricalField,
    "keyword" : CategoricalField,
    "text" : WordSequenceField,
    "relationship" : RelationshipField,
    #"image" : field.ImageField,
    #"video" : field.VideoField,
    #"audio" : field.AudioField,
    "id" : IdField,
    "entity_type" : EntityTypeField,
}

from starcoder.field_encoder import FieldEncoder, NumericEncoder, CategoricalEncoder, SequenceEncoder
from starcoder.field_decoder import FieldDecoder, NumericDecoder, CategoricalDecoder, SequenceDecoder
from starcoder.field_loss import FieldLoss, NumericLoss, CategoricalLoss, SequenceLoss
field_model_classes: Dict[Type[DataField], Tuple[Type[FieldEncoder], Type[FieldDecoder], Type[FieldLoss]]] = { # type: ignore[type-arg]
    NumericField : (NumericEncoder, NumericDecoder, NumericLoss),
    #field.DistributionField : (field_models.DistributionEncoder, field_models.DistributionDecoder, field_models.DistributionLoss),
    #field.IntegerField : (field_models.NumericEncoder, field_models.NumericDecoder, field_models.NumericLoss),
    CategoricalField : (CategoricalEncoder, CategoricalDecoder, CategoricalLoss),
    CharacterSequenceField : (SequenceEncoder, SequenceDecoder, SequenceLoss),
    WordSequenceField : (SequenceEncoder, SequenceDecoder, SequenceLoss),
    #field.DateField : (field_models.NumericEncoder, field_models.NumericDecoder, field_models.NumericLoss),
    #field.CharacterField : (field_models.TextEncoder, field_models.TextDecoder, field_models.TextLoss),
    #field.ImageField : (field_encoder.ImageEncoder, field_decoder.ImageDecoder, field_loss.ImageLoss),
    #field.VideoField : (field_encoder.VideoEncoder, field_decoder.VideoDecoder, field_loss.VideoLoss),
    #field.AudioField : (field_encoder.AudioEncoder, field_decoder.AudioDecoder, field_loss.AudioLoss),    
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


