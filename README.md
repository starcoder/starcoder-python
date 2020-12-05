# StarCoder: computational intelligence for humanities scholarship

StarCoder is a machine learning framework designed for researchers in fields such as History, Literary Criticism, or Classics who are interested in what cutting-edge neural models can reveal about their objects of study.  It accomplishes this by following several principles:

1. **Focus on data**: the humanist need only worry about how they *represent* their material, which is a critical aspect of empirical studies, computational or otherwise.  By using [JSON-LD](https://json-ld.org) to describe [entities and their relationships](https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model), StarCoder encourages humanists to produce completely general, transparent, and explicit archives.
2. **Minimal dependencies**: StarCoder is built from the ground up in [PyTorch](https://pytorch.org), its only dependency other than a recent version of Python.
3. **Unsupervised**
4. **Flexible training**
5. **Interpretable output**

At the same time, StarCoder is also designed for machine learning researchers 


StarCoder's goal is to programmatically generate, train, and employ neural models tailored to complex data sets, thus allowing experts in other fields to remain focused on their particular domain, while benefiting from advancements in machine learning.  StarCoder models can be used for supervised and unsupervised tasks, such as classification, augmentation, cleaning, clustering, anomaly detection, and so forth.  It assumes a typed [Entity-relationship model](https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model) specified in human-readable [JSON conventions](https://json-ld.org/).  StarCoder combines graph-convolutional networks, autoencoders, and an open set of encoder-decoder pairs.  In brief, the procedure is:

1.  For each property, create an *encoder* that translates its type into a fixed-length representation
2.  For each entity-type, create an initial autoencoder for the concatenated output of its properties' representations
3.  Stack additional autoencoder layers that include bottlenecks from adjacent entities in the previous layer
4.  Again for each property, create a *decoder* that translates the autoencoder output back into the property's type

In the simplest case, the model is trained on end-to-end reconstruction loss, and can be used as a similarity measure (e.g. cosine between bottleneck representations), for anomaly detection (e.g. entropy of softmax outputs), or to dynamically explore relationships (e.g. dynamically editing properties).  My selectively masking properties, it can also be trained as a classifier, and a variety of dropout and perturbations can be applied to fit different needs.

See [the tutorial](TUTORIAL.md) for thorough examples of using StarCoder.  As a quick intro to what is required from the domain experts, consider someone studying an email archive.  Two files are required, both in JSON format: a "schema", and a list of entities.  For the email example, the schema might be:

```
{
  "meta" : {
	"entity_type_property" : {"type" : "entity_type"},
	"id_property" : {"type" : "id"}
  },
  "properties" : {
    "content" : {"type" : "text"},
    "date" : {"type" : "date"},
    "name" : {"type" : "text"},
    "role" : {"type" : "categorical"},
    "age" : {"type" : "scalar"}
  },
  "entity_types" : {
    "person" : {
	  "properties" : ["name", "role", "age"]
	},
	"email" : {
	  "properties" : ["date", "content"]
	}
  },
  "relationships" : {
    "sent_by" : {
      "source_entity_type" : "email",
      "target_entity_type" : "person"}
	},
    "received_by" : {
      "source_entity_type" : "email",
      "target_entity_type" : "person"
	}
  }
}
```

while the list of entities might be:

```
[
  {"id" : "1", "entity_type" : "person", "person_name", "age" : 30, "name" : "Chris", "role" : "supervisor"},
  {"id" : "2", "entity_type" : "person", "person_name", "age" : 20, "name" : "Lynn", "role" : "employee"},
  {"id" : "3", "entity_type" : "email", "sent_by" : "1", "received_by" : "2", "date" : "23-3-2020", "content" : "Get back to work!"},
  ...
]
```

StarCoder uses this format directly, but also includes adapters for common formats, particularly tabular (CSV) data.

## Representational conventions

In all cases, tensor dimensions are ordered by increasing specificity.  Admittedly this is somewhat vague and interpretive, but a few concrete examples:

1.  The first dimension *always* indexes the batch
2.  If the property type is grid-like (sequence, image, etc), dimensions that indicate the location always *immediately follow* the batch: for example, a sequence of word-embeddings will have shape `(BATCH_SIZE x SEQUENCE_LENGTH x EMBEDDING_SIZE)`, an RGB image will have shape `(BATCH_SIZE x WIDTH x HEIGHT x 3)`, etc.


As the image example shows, somewhat-arbitrary choices are needed (`HEIGHT` could have preceded `WIDTH`), and these will be documented as they arise and new property types are defined.  *It is very likely that StarCoder will switch to named tensor dimensions and this will become moot.*

## Next steps

See [the todo list](TODO.md) for work that is planned or under-way.

