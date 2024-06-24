<img src="docs/_static/pyrocs-icon-1.png" width="400"/>

# pyRoCS
[![test](https://github.com/sandialabs/pyrocs/actions/workflows/test.yml/badge.svg)](https://github.com/sandialabs/pyrocs/actions/workflows/test.yml)

pyRoCS is a library of functions used to support resilience analysis of complex systems. The package contains example datasets and tutorial to help demonstrate how the functions can be used. 

## Installation
pyRoCS can be installed using `pip`

    pip install pyrocs

Alternatively, pyRoCS can be installed through GitHub

    git clone https://github.com/sandialabs/pyrocs.git
    cd pyrocs
    pip install .

## Tutorials
To get started with pyRoCS, we recommend working through the [tutorials](https://sandialabs.github.io/pyrocs/tutorials.html).


## Package Layout and Documentation
```
docs
pyrocs/
├── biosciences/
│   ├── affinity
│   ├── functional_redundancy
│   └── hill_diversity
├── complex_systems/
│   ├── causal_complexity
│   ├── fluctuation_complexity
│   └── grc
└── information_theory/
    ├── entropy
    ├── kl_divergence
    └── mutual_info
test
tutorials
```

## Citing
A peer-reviewed paper is in progress. So stay tuned for the DOI.

## Contributing
See the [contributing](https://sandialabs.github.io/pyrocs/contributing.html) page for more info

## Copyright and License
pyRoCS is licensed through National Technology & Engineering Solutions of Sandia, LLC (NTESS) under a Revised BSD-3 clause. See license file for more information.
