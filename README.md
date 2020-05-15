# Slavery manifest and fugitive data

## Data model

```mermaid
graph TD;
  Voyage[Voyage]
  Notice[Notice]
  Event[Event]
  Vessel[Vessel]
  Gazette[Gazette]
  Slave[Slave]
  Owner[Owner]
  Consigner[Consigner]
  Shipper[Shipper]
  Subscriber[Subscriber]
  Location[Location]
  
  Location-->Owner
  Location-->Voyage
  Location-->Vessel
  Location-->Consigner
  Location-->Shipper
  Location-->Voyage
  Owner-->Slave
  Slave-->Voyage
  Captain-->Voyage
  Vessel-->Voyage
  Consigner-->Voyage
  Shipper-->Voyage
  Slave-->Event
  Event-->Notice
  Gazette-->Notice
  Subscriber-->Notice
  classDef Notices fill:#090,stroke:#fff;
  classDef Manifests fill:#009,stroke:#fff;
  class Notice,Escape,Gazette,Subscriber,Event Notices;
  class Voyage,Vessel,Cosigner,Shipper,Captain Manifests;
  class Slave,Owner Other;
```

## ElasticSearch infrastructure

## Autoencoding Graph Ensemble (AGE)

```mermaid
graph TD;
  Voyage[Voyage]
  Notice[Notice]
  Event[Event]
  Vessel[Vessel]
  Gazette[Gazette]
  Slave[Slave]
  Owner[Owner]
  Consigner[Consigner]
  Shipper[Shipper]
  Subscriber[Subscriber]
  Location[Location]
  
  Location-->Owner
  Location-->Voyage
  Location-->Vessel
  Location-->Consigner
  Location-->Shipper
  Location-->Voyage
  Owner-->Slave
  Slave-->Voyage
  Captain-->Voyage
  Vessel-->Voyage
  Consigner-->Voyage
  Shipper-->Voyage
  Slave-->Event
  Event-->Notice
  Gazette-->Notice
  Subscriber-->Notice
  classDef Notices fill:#090,stroke:#fff;
  classDef Manifests fill:#009,stroke:#fff;
  class Notice,Escape,Gazette,Subscriber,Event Notices;
  class Voyage,Vessel,Cosigner,Shipper,Captain Manifests;
  class Slave,Owner Other;
```

## Evaluation

After a model is trained, it is evaluated on unseen test data in the following ways: 

  for each field F and instance I, mask I[F] (entity-level field reconstruction)
  for each field F, mask F across all instances (type-level field reconstruction)
  for each edge type (S->T), remove X% of edges, rank potential Ts for P-at-N (edge prediction)
  
