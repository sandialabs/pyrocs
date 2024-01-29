.. _development:

Development
=====================

Installation
------------
To maintain a local installation, developers should use the following commands::
    
    git clone https://github.com/sandialabs/pyrocs.git
    cd pyrocs
    pip install -e .

Testing
-------
To test locally, run::

    pytest pyrocs

at the root of the repository. Note that this requires the installation
of pytest.

Documentation
------------------

Building docs
^^^^^^^^^^^^^^^
To build docs locally, navigate to ``pyrocs/docs`` and run::

    make html

After building, the static html files can be found in ``_build/html``.

Docstrings
^^^^^^^^^^^
The pyrocs documentation adheres to the Google style for docstrings. Not only does this
help to keep a consistent style, but it is also necessary for the API documentation
to be parsed and displayed correctly. Examples can be found in the 
`sphinx documentation <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`_.
