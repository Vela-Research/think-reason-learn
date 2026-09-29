.. Think Reason Learn documentation master file, created by
   sphinx-quickstart on Wed Sep 17 15:37:24 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation
=============

.. meta::
   :description: Documentation for Think, Reason, Learn, Vela Research's open-source Python library for prediction models that show their reasoning.

Think, Reason, Learn is an open-source Python library for prediction models that show their reasoning. It keeps
the shape of decision trees, rule forests and short policies, and asks a language model to reason at each step, so
every prediction comes with the questions or rules behind it. It is built by Vela Research, the research arm of Vela
Partners, with the University of Oxford.

Start with the installation guide, then the quick start, which fits a small GPTree on six example founder profiles.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Get started

      - :doc:`About <getting_started/about>`
      - :doc:`Installation guide <getting_started/installation_guide>`
      - :doc:`Quick start <getting_started/quick_start>`

   .. grid-item-card:: Reference

      - :doc:`API reference <modules>`

   .. grid-item-card:: Project

      - :doc:`Contributing <contributing>`
      - :doc:`Changelog <changelog>`
      - :doc:`License <license>`

The papers behind the methods are on the `research page <https://thinkreasonlearn.com/research>`_.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/about
   getting_started/installation_guide
   getting_started/quick_start
   modules
   contributing
   changelog
   license
