.. _index:

.. image:: _static/pyrocs-icon-1.png
    :width: 400

Overview
============
The purpose of pyRoCS is to support characterization, evaluation, and learning to inform 
improvements in complex systems. Specifically, the focus on resilience draws attention 
to the ability of a complex system to withstand, operate through, and recover from a 
disruption. The complex system may be a physical system such as an electric grid, 
an organization such as a company, or even a subfunction of an organization. Existing 
mathematical equations for resilience analysis are found within multiple domains including 
information theory, biological sciences, and complex systems. This package synthesizes and 
refactors equations from these various domains to generalize their implementations in a 
Python environment. 

To get started, we recommend exploring the pyrocs :ref:`tutorials<tutorials>`.

Datasets can be analyzed in multiple ways using using functions within pyRoCS. 
While most of these datasets are in array formats, they reflect different underlying 
structures, with some representing counts of entries and
sequences and other representing summaries of graph-based structures. The below table summarizes
current functionality and their corresponding data inputs organized by module.

.. list-table:: Modules, Functions and Primary Data Inputs
   :widths: 25 25 75
   :header-rows: 1

   * - Module
     - Function
     - Primary Data Inputs
   * - Information Theory 
     - discrete_entropy
     - Unique entries in a process and associated counts
   * - Information Theory 
     - kl_divergence
     - Two arrays of probability distributions
   * - Information Theory 
     - novelty_transience_resonance
     - An array of probability distribution
   * - Information Theory 
     - mutual_info
     - Two arrays of entries (optional: counts)
   * - Biosciences
     - affinity
     - Matrix of co-occurring variables (optional: weights)
   * - Biosciences
     - functional_redundancy
     - Array of relative abundance and array of symmetric similarities
   * - Biosciences
     - hill_diversity
     - Proportion of individuals in group
   * - Biosciences
     - hill_shannon
     - Proportion of individuals in group
   * - Biosciences
     - hill_simpson
     - Proportion of individuals in group
   * - Complex Systems
     - causal_complexity
     - Adjacency matrix of graph structure
   * - Complex Systems
     - cyclomatic_complexity
     - Adjacency matrix of graph structure
   * - Complex Systems
     - feedback_density
     - Adjacency matrix of graph structure
   * - Complex Ssystems
     - fluctuation_complexity
     - Array of sequenced events
   * - Complex Systems
     - grc
     - Adjacency matrix of graph structure

The :ref:`API documentation<apidoc>` provides additional details about the functions 
included in the package.

.. toctree::
    :maxdepth: 1
    :hidden:
    
    Overview <self>
    installation
    tutorials
    contributing
    development
    apidoc
    references
