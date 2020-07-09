from starcoder.fields import NumericField, DistributionField, CategoricalField, SequentialField, IntegerField, RelationField, DateField, IdField, EntityTypeField

from starcoder.models import NumericEncoder, NumericDecoder, NumericLoss, DistributionEncoder, DistributionDecoder, DistributionLoss, CategoricalEncoder, CategoricalDecoder, CategoricalLoss, SequentialEncoder, SequentialDecoder, SequentialLoss

from starcoder.schedulers import Scheduler
from starcoder import batchifiers
from starcoder import splitters
from starcoder import fields
# import RandomEntities, RandomComponents, DuplicateEntityTypes

field_model_classes = {
    NumericField : (NumericEncoder, NumericDecoder, NumericLoss),
    DistributionField : (DistributionEncoder, DistributionDecoder, DistributionLoss),
    IntegerField : (NumericEncoder, NumericDecoder, NumericLoss),
    CategoricalField : (CategoricalEncoder, CategoricalDecoder, CategoricalLoss),
    SequentialField : (SequentialEncoder, SequentialDecoder, SequentialLoss),
    #WordField : (SequentialEncoder, SequentialDecoder, SequentialLoss),
    DateField : (NumericEncoder, NumericDecoder, NumericLoss),
    fields.CharacterField : (SequentialEncoder, SequentialDecoder, SequentialLoss),
}

summarizer_classes = {}

projector_classes = {}

batchifier_classes = {"sample_entities" : batchifiers.SampleEntities,
                      "sample_snowflakes" : batchifiers.SampleSnowflakes,
                      "sample_components" : batchifiers.SampleComponents,
}

scheduler_classes = {"default" : Scheduler}

splitter_classes = {"sample_entities" : splitters.SampleEntities,
                    "sample_components" : splitters.SampleComponents,
}

field_classes = {"numeric" : NumericField,
                 "categorical" : CategoricalField,
                 "boolean" : CategoricalField,                 
                 "sequential" : SequentialField,
                 "integer" : IntegerField,
                 "keyword" : CategoricalField,
                 "text" : fields.CharacterField,
                 "relation" : RelationField,
                 "distribution" : DistributionField,
                 "date" : DateField,
                 "id" : IdField,
                 "entity_type" : EntityTypeField,
}


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()
