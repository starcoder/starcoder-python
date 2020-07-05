#!/usr/bin/env python

from distutils.core import setup

setup(name="starcoder",
      version="0.0.2",
      description="StarCoder is an unsupervised deep learning model tailored for exploratory empirical research in the humanities.",
      long_description="StarCoder is essentially a generator that combines autoencoder and graph-convolutional mechanisms with the open set of neural architectures to build end-to-end models of entity-relationship schemas.  By adopting intuitive JSON for all I/O, and using reconstruction loss as the objective, it allows researchers from other fields to apply it to their specific areas, while ML researchers can continue to augment it with new techniques.",
      long_description_content_type="text/x-rst",
      author="Thomas Lippincott",
      author_email="tom@cs.jhu.edu",
      url="https://github.com/TomLippincott/starcoder",
      maintainer="Thomas Lippincott",
      maintainer_email="tom@cs.jhu.edu",
      packages=["starcoder"],
      scripts=[],
      requires=["torch", "pytest", "sklearn", "scipy"],
     )
