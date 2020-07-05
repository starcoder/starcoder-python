# StarCoder Overview

StarCoder's goal is to programmatically generate, train, and employ neural models tailored to complex data sets, thus allowing experts in other fields to remain focused on their particular domain, while benefiting from advancements in machine learning.  StarCoder models can be used for supervised and unsupervised tasks, such as classification, augmentation, cleaning, clustering, anomaly detection, and so forth.  It assumes a typed [Entity-relationship model](https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model) specified in human-readable [JSON conventions](https://json-ld.org/).  StarCoder combines graph-convolutional networks, autoencoders, and an open set of encoder-decoder pairs.  In brief, the procedure is:

1.  For each field, create an *encoder* that translates its type into a fixed-length representation
2.  For each entity-type, create an initial autoencoder for the concatenated output of its fields' representations
3.  Stack additional autoencoder layers that include bottlenecks from adjacent entities in the previous layer
4.  Again for each field, create a *decoder* that translates the autoencoder output back into the field's type

In the simplest case, the model is trained on end-to-end reconstruction loss, and can be used as a similarity measure (e.g. cosine between bottleneck representations), for anomaly detection (e.g. entropy of softmax outputs), or to dynamically explore relationships (e.g. dynamically editing fields).  My selectively masking fields, it can also be trained as a classifier, and a variety of dropout and perturbations can be applied to fit different needs.

See [the tutorial](TUTORIAL.md) for thorough examples of using StarCoder.  As a quick intro to what is required from the domain experts, consider someone studying an email archive.  Two files are required, both in JSON format: a "schema", and a list of entities.  For the email example, the schema might be:

```
{
  "entity_type_field" : {"type" : "entity_type"},
  "id_field" : {"type" : "id"},
  "content" : {"type" : "text"},
  "date" : {"type" : "date"},
  "name" : {"type" : "text"},
  "role" : {"type" : "categorical"},
  "age" : {"type" : "numeric"}
  "sent_by" : {"type" : "relation",
	           "source_entity_type" : "email",
			   "target_entity_type" : "person"},
  "received_by" : {"type" : "relation",
                   "source_entity_type" : "email",
	               "target_entity_type" : "person"}
}
```

while the list of entities might be:

```
[
  {"id" : "1", "entity_type_field" : "person", "person_name", "age" : 30, "name" : "Chris", "role" : "supervisor"},
  {"id" : "2", "entity_type_field" : "person", "person_name", "age" : 20, "name" : "Lynn", "role" : "employee"},
  {"id" : "3", "entity_type_field" : "email", "sent_by" : "1", "received_by" : "2", "date" : "23-3-2020", "content" : "Get back to work!"},
  ...
]
```

StarCoder uses this format directly, but also includes adapters for common formats, particularly tabular (CSV) data.

## Representational conventions

In all cases, tensor dimensions are ordered by increasing specificity.  Admittedly this is somewhat vague and interpretive, but a few concrete examples:

1.  The first dimension *always* indexes the batch
2.  If the field type is grid-like (sequence, image, etc), dimensions that indicate the location always *immediately follow* the batch: for example, a sequence of word-embeddings will have shape `(BATCH_SIZE x SEQUENCE_LENGTH x EMBEDDING_SIZE)`, an RGB image will have shape `(BATCH_SIZE x WIDTH x HEIGHT x 3)`, etc.


As the image example shows, somewhat-arbitrary choices are needed (`HEIGHT` could have preceded `WIDTH`), and these will be documented as they arise and new field types are defined.  *It is very likely that StarCoder will switch to named tensor dimensions and this will become moot.*

## Next steps

See [the todo list](TODO.md) for work that is planned or under-way.

