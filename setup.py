from setuptools import setup

DESCRIPTION = ('pyrocs is a python library to support the resilience analysis of complex systems.')

LONG_DESCRIPTION = """
pyrocs is a python package for resilience analysis of complex systems. 
It is a collection of functions drawn from multiple disciplines (including information theory, biosciences, and complex systems) 
that can be used to support characterization of a complex system's ability to withstand, operate through, and recover from disruptions. 
The library includes functions that can be used to evaluate representative nature of datasets as well as characterize structures of different organizations.

Documentation: https://sandialabs.github.io/pyrocs/tutorials.html

Source code: https://github.com/sandialabs/pyrocs
"""

DISTNAME = 'pyrocs'
MAINTAINER = "Thushara Gunda"
MAINTAINER_EMAIL = 'tgunda@sandia.gov'
AUTHOR = 'pyRoCS Developers'
LICENSE = 'BSD 3-Clause License'
URL = 'https://github.com/sandialabs/pyrocs'

setup(
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    url=URL
)
