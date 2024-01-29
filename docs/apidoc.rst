.. _api-doc:

API Documentation
==================
Currently, the pyRoCS package includes mathematical formulations of equations from three domains: 
information theory, biosciences, and complex systems. These domains have advanced how we quantify, 
store, and communicate information (Information theory); explore interactions between species (biosciences); 
and consider non-linear interactions between system components (complex systems). 

Functions within pyRoCS are organized into modules that reflect the disciplines from which they 
originate. While they have distinct origins, the functions can also be applied to resilience 
analysis of any complex system. Accordingly, their formulations and associated documentation 
have been generalized in this package. Each of the function's documentation includes information 
about the original publication used to develop the calculation; see :ref:`references<references>` more details. 
In addition to converting these equations from the literature into formal Python 
implementations, the calculations were also modified to support usability 
(e.g., exposing more inputs to the user and diversifying the types of data 
structures accepted by the calculations). 

See the links below for more details about the functions captured within each of the modules:


.. toctree::
   :maxdepth: 1

   apidoc-pages/biosciences
   apidoc-pages/complex_systems
   apidoc-pages/information_theory
