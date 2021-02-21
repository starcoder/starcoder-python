import torch
import numpy
from torch.utils.data import DataLoader, Dataset
import logging
from abc import ABCMeta, abstractproperty, abstractmethod
from starcoder.base import StarcoderObject
from starcoder.utils import starport
import torch.autograd.profiler as profiler

logger = logging.getLogger(__name__)


class Ensemble(StarcoderObject, torch.nn.Module, metaclass=ABCMeta):

    def __init__(self) -> None:
        super(Ensemble, self).__init__()

    def forward(self, entities, adjacencies):
        with profiler.record_function("AUTOENCODER"):
            return self._forward(entities, adjacencies)

    @abstractmethod
    def _forward(self, x): pass


class GraphAutoencoder(Ensemble):
    
    def __init__(self,
                 schema,
                 data,
                 device=torch.device("cpu")
             ):
        """
        """
        super(GraphAutoencoder, self).__init__()
        self.schema = schema
        self.reverse_relationships = self.schema["meta"]["reverse_relationships"]
        self.depth = self.schema["meta"]["depth"]
        self.depthwise_boost = "highway"
        self.device = device
        projected_size = None
        # TODO: limit depth to longest path through network
        # A class for each property that can manipulate its human and computer forms
        self.property_objects = {}
        for property_name, property_spec in self.schema["properties"].items():
            property_class = starport(property_spec["meta"]["class"])
            self.property_objects[property_name] = property_class(property_name, property_spec, data)
        # An encoder for each property that turns its data type into a fixed-size representation
        property_encoders = {}
        for property_name, property_spec in self.schema["properties"].items():
            property_encoder_class = starport(property_spec["meta"]["encoder"])
            property_encoders[property_name] = property_encoder_class(
                self.property_objects[property_name],
                **property_spec.get("meta", {})
            )
        self.property_encoders = torch.nn.ModuleDict(property_encoders)
        # The size of an encoded entity is the sum of the base representation size, the encoded
        # sizes of its possible properties, and a binary indicator for the presence of each property.
        self.boundary_sizes = {}
        self.encoded_sizes = {}
        for entity_type_name, entity_type_spec in self.schema["entity_types"].items():
            self.boundary_sizes[entity_type_name] = 0
            self.encoded_sizes[entity_type_name] = 0
            for property_name in entity_type_spec["properties"]:
                self.encoded_sizes[entity_type_name] += self.property_encoders[property_name].output_size
                self.boundary_sizes[entity_type_name] += self.property_encoders[property_name].output_size
        # An autoencoder for each entity type and depth
        # The first has input layers of size equal to the size of the corresponding entity's
        # representation size and a fixed-size output.  The rest have input layers of the 
        # previous output size plus a bottleneck size for each possible (forward or reverse) 
        # relationship.
        self.entity_autoencoders = {}
        self.entity_bottlenecks_combined_sizes = {}
        activation = torch.nn.functional.relu
        null_autoencoder_class = starport(schema["meta"]["null_autoencoder"])
        autoencoder_class = starport(schema["meta"]["autoencoder"])
        for depth in range(self.depth + 1):
            for entity_type_name, entity_type in schema["entity_types"].items():
                self.entity_autoencoders[entity_type_name] = self.entity_autoencoders.get(
                    entity_type_name,
                    []
                )
                input_size = (
                    self.encoded_sizes[entity_type_name] 
                    if depth == 0 
                    else self.entity_autoencoders[entity_type_name][depth - 1].output_size
                )
                if depth > 0:
                    relationship_size = 0
                    for rel_name, rel in schema.get("relationships", {}).items():
                        s_et = rel["source_entity_type"]
                        t_et = rel["target_entity_type"]
                        if s_et == entity_type_name:
                            input_size += self.entity_autoencoders[t_et][depth - 1].bottleneck_size
                        if t_et == entity_type_name:
                            input_size += self.entity_autoencoders[s_et][depth - 1].bottleneck_size
                autoencoder_shape = entity_type["meta"]["autoencoder_shape"]
                bottleneck_size = entity_type["meta"]["bottleneck_size"]
                self.entity_bottlenecks_combined_sizes[entity_type_name] = (
                    self.entity_bottlenecks_combined_sizes.get(
                        entity_type_name,
                        0
                    )
                )
                self.entity_bottlenecks_combined_sizes[entity_type_name] += bottleneck_size
                self.entity_autoencoders[entity_type_name].append(
                    null_autoencoder_class(entity_type_name, depth) if input_size == 0 else autoencoder_class(
                        entity_type_name,
                        depth,
                        input_size,
                        bottleneck_size,
                        autoencoder_shape, 
                        activation
                    )
                )
        self.entity_autoencoders = torch.nn.ModuleDict(
            {k : torch.nn.ModuleList(v) for k, v in self.entity_autoencoders.items()}
        )
        self.entity_normalizers = torch.nn.ModuleDict(
            {k : starport(self.schema["entity_types"][k]["meta"]["normalizer"])(k, v[0].input_size) for k, v in self.entity_autoencoders.items()}
        )
        # A summarizer for each relationship particant (source or target), at each
        # depth less than the max, to reduce one-to-many relationships to a fixed size.
        relationship_source_summarizers = {}
        relationship_target_summarizers = {}
        for d in range(self.depth):
            for rel_name, rel in schema.get("relationships", {}).items(): 
                st = rel["source_entity_type"]
                tt = rel["target_entity_type"]
                relationship_source_summarizers[rel_name] = relationship_source_summarizers.get(rel_name, [])
                relationship_target_summarizers[rel_name] = relationship_target_summarizers.get(rel_name, [])
                sbs = self.entity_autoencoders[st][d].bottleneck_size
                tbs = self.entity_autoencoders[tt][d].bottleneck_size
                relationship_source_summarizers[rel_name].append(
                    starport(rel["meta"]["source_summarizer"] if sbs > 0 else "starcoder.summarizer.NullSummarizer")(
                        rel_name, 
                        "source",
                        sbs
                    )
                )
                relationship_target_summarizers[rel_name].append(
                    starport(rel["meta"]["target_summarizer"] if tbs > 0 else "starcoder.summarizer.NullSummarizer")(
                        rel_name, 
                        "target",
                        tbs
                    )
                )
        self.relationship_source_summarizers = torch.nn.ModuleDict(
            {k : torch.nn.ModuleList(v) for k, v in relationship_source_summarizers.items()}
        )
        self.relationship_target_summarizers = torch.nn.ModuleDict(
            {k : torch.nn.ModuleList(v) for k, v in relationship_target_summarizers.items()}
        )
        # A binary classifier for each relationship at each autoencoder depth, using bottlenecks
        relationship_detectors = {}
        for relationship_name, relationship_spec in self.schema["relationships"].items():
            continue
            st = relationship_spec["source_entity_type"]
            tt = relationship_spec["target_entity_type"]
            relationship_detectors[relationship_name] = starport(
                relationship_spec["meta"]["relationship_detector"]
            )(
                self.entity_bottlenecks_combined_sizes[st],
                self.entity_bottlenecks_combined_sizes[tt],
            )
        self.relationship_detectors = torch.nn.ModuleDict(relationship_detectors)
        # MLP for each entity type to project representations to a common size
        self.projected_size = projected_size if projected_size != None else max(self.boundary_sizes.values()) * (self.depth + 1)
        # actually only need to project entities if/when they share properties...
        projectors = {}
        projector_class = starport(self.schema["meta"]["projector"])
        for entity_type_name in self.schema["entity_types"].keys():
            boundary_size = 0
            for ae in self.entity_autoencoders[entity_type_name]:
                boundary_size += ae.output_size
            projectors[entity_type_name] = projector_class(entity_type_name, boundary_size, self.projected_size, activation) if boundary_size not in [0, self.projected_size] else torch.nn.Identity()
        self.projectors = torch.nn.ModuleDict(projectors)
        # A decoder for each property that takes a projected representation 
        # and generates a value of the property's data type
        property_decoders = {}
        for property_name, property_spec in self.schema["properties"].items():
            property_decoder_class = starport(property_spec["meta"]["decoder"])
            property_decoders[property_name] = property_decoder_class(
                self.property_objects[property_name],
                self.projected_size,
                activation,
                encoder=self.property_encoders[property_name]
            )
        self.property_decoders = torch.nn.ModuleDict(property_decoders)

        # A way to calculate loss for each property
        self.property_losses = {}
        for property_name, property_spec in self.schema["properties"].items():
            property_loss_class = starport(property_spec["meta"]["loss"])
            self.property_losses[property_name] = property_loss_class(
                self.property_objects[property_name]
            )

    def cuda(self, device="cuda:0"):
        self.device = torch.device(device)
        return super(GraphAutoencoder, self).cuda()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def pack_properties(self, entities):        
        entity_type_to_batch_indices = {et : [] for et in self.schema["entity_types"].keys()}
        entity_type_to_property_indices = {et : {p : [] for p in self.schema["properties"].keys()} for et in self.schema["entity_types"].keys()}
        logger.debug("Assembling entity, property, and (entity, property) indices")        
        packed_properties = {}
        for bi, entity in enumerate(entities):
            entity_type_name = entity[self.schema["entity_type_property"]]
            ei = len(entity_type_to_batch_indices[entity_type_name])
            entity_type_to_batch_indices[entity_type_name].append([ei, bi])
            for property_name in self.schema["entity_types"][entity_type_name]["properties"]: 
                packed_properties[property_name] = packed_properties.get(property_name, [])
                property_object = self.property_objects[property_name]
                entity_type_to_property_indices[entity_type_name] = entity_type_to_property_indices.get(
                    entity_type_name, 
                    {}
                )
                entity_type_to_property_indices[entity_type_name][property_name] = (
                    entity_type_to_property_indices[entity_type_name].get(property_name, [])
                )
                if property_name not in entity:
                    continue
                packed_property_value = property_object.pack(entity[property_name])                
                pi = len(packed_properties[property_name])
                packed_properties[property_name].append(packed_property_value)
                entity_type_to_property_indices[entity_type_name][property_name].append(
                    [ei, pi]
                )
        for property_name in list(packed_properties.keys()):
            if len(packed_properties[property_name]) == 0:
                stand_in = torch.tensor(self.property_objects[property_name].missing_value)
                packed_properties[property_name] = torch.zeros(
                    size=(0,) + stand_in.shape,
                    device=self.device,
                    dtype=self.property_objects[property_name].stacked_type
                )
            else:
                packed_properties[property_name] = torch.tensor(
                    packed_properties[property_name],
                    device=self.device,
                    dtype=self.property_objects[property_name].stacked_type
                )
        for entity_type_name, indices in entity_type_to_batch_indices.items():
            logger.debug("%d '%s' entities", len(indices), entity_type_name)
        return (
            {
                k : torch.tensor(
                    v,
                    dtype=torch.int64,
                    device=self.device
                ).reshape(shape=(-1, 2)) for k, v in entity_type_to_batch_indices.items()
            }, 
            {
                p : {
                    k : torch.tensor(v, dtype=torch.int64, device=self.device).reshape(shape=(-1, 2)) for k, v in vv.items()
                } for p, vv in entity_type_to_property_indices.items()
            }, 
            packed_properties
        )

    def encode_properties(self, packed_properties):
        logger.debug("Encoding each input property to a fixed-length representation")
        encoded_properties = {}
        for property_name, property_values in packed_properties.items():
            encoded_properties[property_name] = self.property_encoders[property_name](
                property_values
            )
        return encoded_properties
    
    def create_autoencoder_inputs(self, 
                                  encoded_properties, 
                                  entity_type_to_batch_indices,
                                  entity_type_to_property_indices):
        logger.debug("Constructing entity-autoencoder inputs by concatenating property encodings")
        autoencoder_inputs = {}
        autoencoder_input_lists = {}
        for entity_type_name, et_to_batch_indices in entity_type_to_batch_indices.items():
            autoencoder_input_lists[entity_type_name] = []
            entity_count = et_to_batch_indices.shape[0]
            for property_name in self.schema["entity_types"][entity_type_name]["properties"]:
                et_to_prop_indices = entity_type_to_property_indices.get(entity_type_name, {}).get(property_name)
                vals = torch.zeros(
                    size=(entity_count, self.property_encoders[property_name].output_size),
                    device=self.device
                )
                idx = entity_type_to_property_indices[entity_type_name][property_name]
                if idx.shape[0] != 0:
                    vals[idx[:, 0]] = encoded_properties[property_name][idx[:, 1]]
                autoencoder_input_lists[entity_type_name].append(vals)
            autoencoder_inputs[entity_type_name] = self.entity_normalizers[entity_type_name](
                torch.cat(
                    autoencoder_input_lists[entity_type_name] + [torch.zeros(size=(entity_count, 0), device=self.device)],
                    1
                )
            )
        logger.debug("Shapes: %s", {k : v.shape for k, v in autoencoder_inputs.items()})
        return autoencoder_inputs    

    def run_autoencoder(self,
                        depth,
                        autoencoder_inputs):
        logging.debug("Running depth-%d autoencoder", depth)
        bottlenecks = {}
        autoencoder_outputs = {}
        for entity_type_name, entity_type_spec in self.schema["entity_types"].items():
            entity_reps = autoencoder_inputs.get(entity_type_name, None)
            if entity_reps != None:
                entity_outputs, bns = self.entity_autoencoders[entity_type_name][depth](
                    autoencoder_inputs[entity_type_name]
                )
                if entity_outputs != None:
                    autoencoder_outputs[entity_type_name] = entity_outputs
                if bns != None:
                    bottlenecks[entity_type_name] = bns
        return (bottlenecks, autoencoder_outputs)

    def create_structured_autoencoder_inputs(self,
                                             depth,
                                             prev_outputs,
                                             prev_bottlenecks,
                                             adjacencies):
        autoencoder_inputs = {}
        for entity_type_name, prev_output in prev_outputs.items():
            entity_type = self.schema["entity_types"][entity_type_name]
            autoencoder_inputs[entity_type_name] = [prev_output]
            for rel_name, rel_spec in sorted(self.schema["relationships"].items()):
                source = rel_spec["source_entity_type"]
                target = rel_spec["target_entity_type"]                
                if source == entity_type_name:
                    summarizer = self.relationship_target_summarizers[rel_name][depth - 1]
                    other = target
                    rel_count = torch.cat([adjacencies[rel_name].sum(1), torch.tensor([0], dtype=adjacencies[rel_name].dtype, device=self.device)])
                    max_rel_count = rel_count.max()
                    cur_ent_count = adjacencies[rel_name].shape[0]
                    related = torch.zeros(size=(cur_ent_count,
                                                max_rel_count + 1, # extra to avoid nan
                                                summarizer.input_size
                                                ), device=self.device)
                    for i in range(adjacencies[rel_name].shape[0]):
                        related[i, :rel_count[i]] = prev_bottlenecks[other][adjacencies[rel_name][i, :]]
                    autoencoder_inputs[entity_type_name].append(summarizer(related))
                if rel_spec["target_entity_type"] == entity_type_name:
                    summarizer = self.relationship_source_summarizers[rel_name][depth - 1]
                    other = source
                    rel_count = torch.cat([adjacencies[rel_name].sum(0), torch.tensor([0], dtype=adjacencies[rel_name].dtype, device=self.device)])
                    max_rel_count = rel_count.max()
                    cur_ent_count = adjacencies[rel_name].shape[1]
                    related = torch.zeros(size=(cur_ent_count,
                                                max_rel_count + 1, # extra to avoid nan
                                                summarizer.input_size
                                                ), device=self.device)
                    for i in range(adjacencies[rel_name].shape[1]):
                        related[i, :rel_count[i]] = prev_bottlenecks[other][adjacencies[rel_name][:,i]]
                    autoencoder_inputs[entity_type_name].append(summarizer(related))
        autoencoder_inputs = {k : torch.cat(v, 1) for k, v in autoencoder_inputs.items()}
        return autoencoder_inputs
                    
    def project_autoencoder_outputs(self, autoencoder_outputs):
        resized_autoencoder_outputs = {}        
        for entity_type_name, ae_output in autoencoder_outputs.items():
            resized_autoencoder_outputs[entity_type_name] = self.projectors[entity_type_name](ae_output)
        return resized_autoencoder_outputs
    
    def decode_properties(self, 
                          resized_encoded_entities,
                          entity_type_to_batch_indices,
                          entity_type_to_property_indices,
    ):
        """
        Decode *all possible properties* for each entity, according to its entity type.
        """
        reconstructions = {}
        indices = {}
        for entity_type_name, property_name_to_indices in entity_type_to_property_indices.items():
            for property_name, index_pairs in property_name_to_indices.items():
                indices[property_name] = indices.get(property_name, [])
                reconstructions[property_name] = reconstructions.get(property_name, [])
                indices[property_name].append(index_pairs)
                reconstructions[property_name].append(
                    self.property_decoders[property_name](resized_encoded_entities[entity_type_name])
                )
                logger.debug("Decoded values for property '%s' of entity type '%s' into shape %s",
                             property_name,
                             entity_type_name,
                             reconstructions[property_name][-1].shape[0]
                         )
        indices = {k : torch.cat(v, 0) for k, v in indices.items()}        
        retval = {k : torch.cat(v, 0) for k, v in reconstructions.items()}
        return (indices, retval)
    
    def assemble_entities(self, 
                          decoded_properties, 
                          entity_type_to_batch_indices, 
                          entity_type_to_property_indices,
                          entities):
        num = sum([len(v) for v in entity_indices.values()])
        retval = {
            self.schema["entity_type_property"] : [None] * num,
            self.schema["id_property"] : [None] * num,
        }
        for entity_type_name, properties in decoded_properties.items():
            indices = entity_indices[entity_type_name]
            for property_name, property_values in properties.items():
                retval[property_name] = retval.get(
                    property_name, 
                    torch.empty(
                        size=(num,) + property_values.shape[1:],
                        device=self.device,
                    )
                )
                retval[property_name][indices] = property_values
        retval[self.schema["id_property"]] = numpy.array(
            [e[self.schema["id_property"]] for e in entities]
        )
        retval[self.schema["entity_type_property"]] = numpy.array(
            [e[self.schema["entity_type_property"]] for e in entities]
        )
        return retval

    def prepare_adjacencies(self, adjacencies, entity_type_to_batch_indices):
        retval = {}
        for relationship_name, adj in adjacencies.items():
            source = self.schema["relationships"][relationship_name]["source_entity_type"]
            target = self.schema["relationships"][relationship_name]["target_entity_type"]
            retval[relationship_name] = adj[
                entity_type_to_batch_indices[source][:, 1].flatten().tolist(),
                :
            ][
                :,
                entity_type_to_batch_indices[target][:, 1].flatten().tolist()
            ].to(self.device)
        return retval

    def _forward(self, entities, adjacencies):
        num_entities = len(entities)
        logger.debug("Starting forward pass with %d entities", num_entities)
        entity_type_to_batch_indices, entity_type_to_property_indices, packed_properties = self.pack_properties(
            entities
        )
        adjacencies = self.prepare_adjacencies(adjacencies, entity_type_to_batch_indices)
        encoded_properties = self.encode_properties(packed_properties)
        vv = sum([v.sum() for v in encoded_properties.values()])
        all_bottlenecks = {}
        all_ae_outputs = []        
        for depth in range(self.depth + 1):
            encoded_entities = self.create_autoencoder_inputs(
                encoded_properties,
                entity_type_to_batch_indices,
                entity_type_to_property_indices             
            ) if depth == 0 else self.create_structured_autoencoder_inputs(
                depth,
                all_ae_outputs[-1],
                {k : v[-1] for k, v in all_bottlenecks.items()},
                adjacencies
            )
            bottlenecks, encoded_entities = self.run_autoencoder(
                depth,
                encoded_entities
            )
            for entity_type_name, b in bottlenecks.items():
                all_bottlenecks[entity_type_name] = all_bottlenecks.get(entity_type_name, [])
                all_bottlenecks[entity_type_name].append(b)
            all_ae_outputs.append(encoded_entities)
        encoded_entities = {}
        for entity_type_name in list(all_ae_outputs[0].keys()):
            encoded_entities[entity_type_name] = torch.cat(
                [out[entity_type_name] for out in all_ae_outputs],
                1
            )
        resized_encoded_entities = self.project_autoencoder_outputs(encoded_entities)
        indices, decoded_properties = self.decode_properties(
            resized_encoded_entities,
            entity_type_to_batch_indices,
            entity_type_to_property_indices
        )
        property_loss = self.compute_property_losses(
            packed_properties, 
            {k : v[indices[k][:, 0]] for k, v in decoded_properties.items()}
        )
        relationship_loss = 0.0
        #self.compute_relationship_losses(
        #    adjacencies,
        #    {k : torch.cat(v, 1) for k, v in all_bottlenecks.items()},
        #)
        reconstructions_by_id = {} if self.training else self.reconstruct_entities(
            decoded_properties, 
            entities, 
            entity_type_to_batch_indices,
        )
        bottlenecks_by_id = self.assemble_bottlenecks(
            {k : v[-1] for k, v in all_bottlenecks.items()},
            entities, 
            entity_type_to_batch_indices
        )
        logger.debug("Returning losses, reconstructions, and bottlenecks")
        return (entities, property_loss + relationship_loss, reconstructions_by_id, bottlenecks_by_id)

    def compute_relationship_losses(self, adjacencies, bottlenecks):
        val = 0.0
        for relationship_name, adj in adjacencies.items():
            st = self.schema["relationships"][relationship_name]["source_entity_type"]
            tt = self.schema["relationships"][relationship_name]["target_entity_type"]
            det = self.relationship_detectors[relationship_name]
            pos = torch.nonzero(adj)
            neg = torch.nonzero(adj == False)
            # will be problem for super-dense graphs...
            neg = neg[torch.randint(neg.shape[0], size=(pos.shape[0],))]
            pairs = torch.cat([pos, neg], 0)
            x = torch.cat(
                [
                    torch.index_select(
                        bottlenecks[st],
                        0,
                        pairs[:, 0]
                    ),
                    torch.index_select(
                        bottlenecks[tt],
                        0,
                        pairs[:, 1]
                    )
                ],
                1
            )
            y = torch.cat(
                [
                    torch.ones(size=(neg.shape[0],), dtype=torch.int64, device=self.device), 
                    torch.zeros(size=(neg.shape[0],), dtype=torch.int64, device=self.device)
                ], 
                0
            )
            val += torch.nn.functional.cross_entropy(det(x), y, reduction="mean")
        return val

    def compute_property_losses(self, packed_entities, decoded_properties):
        logger.debug("Started computing all losses")
        losses_by_property = {}
        for property_name, gold_values in packed_entities.items():
            reconstructed_values = decoded_properties[property_name]
            losses_by_property[property_name] = torch.nansum(
                self.property_losses[property_name](
                    reconstructed_values,
                    gold_values
                )
            )
        logger.debug("Finished computing all losses")

        return torch.sum(torch.cat([torch.nansum(v).flatten() for k, v in losses_by_property.items()]))

    def assemble_bottlenecks(self, bottlenecks, entities, entity_type_to_batch_indices):
        bottlenecks_by_id = {}
        for entity_type_name, bns in bottlenecks.items():
            entity_indices = entity_type_to_batch_indices[entity_type_name][:, 1].cpu()
            ids = [entities[i][self.schema["id_property"]] for i in entity_indices]
            # strange how numpy.array can be a scalar here!
            if isinstance(ids, str):
                ids = [ids]
            for i, bn in zip(ids, bns):
                bottlenecks_by_id[i] = bn.cpu().detach().tolist()
        return bottlenecks_by_id

    # Recursively initialize model weights
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv1d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # performance: cuda to cpu conversion
    def reconstruct_entities(self, 
                             decoded_properties, 
                             entities, 
                             entity_type_to_batch_indices, 
    ):
        retval = {}
        entity_lookups = {
            e : {
                bi : ei for ei, bi in v.cpu().tolist()
            } for e, v in entity_type_to_batch_indices.items()
        }
        for i, entity in enumerate(entities):
            entity_type_name = entity[self.schema["entity_type_property"]]
            reconstruction = {
                self.schema["id_property"] : entity[self.schema["id_property"]],
                self.schema["entity_type_property"] : entity[self.schema["entity_type_property"]],
            }
            for property_name in self.schema["entity_types"][
                    reconstruction[self.schema["entity_type_property"]]]["properties"]:
                reconstruction[property_name] = self.property_objects[property_name].unpack(
                    self.property_losses[property_name].normalize(
                        decoded_properties[property_name][entity_lookups[entity_type_name][i]].cpu().detach()
                    )
                )
            retval[reconstruction[self.schema["id_property"]]] = reconstruction
        return retval
