.. pynol documentation master file, created by
   sphinx-quickstart on Mon Jun 27 22:08:31 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyNOL!
=================

PyNOL is a **Py**\ thon package for **N**\ on-stationary **O**\ nline **L**\earning.

The purpose of this package is to provide a general framework to implement
various algorithms designed for online learning in non-stationary environments.
In particular, we pay special attention to those online algorithms that provably
optimize the *dynamic regret* or *adaptive regret*, two widely used performance
metrics for non-stationary online learning.

There are various algorithms devised to optimize the measures (dynamic regret or
adaptive regret) during the decades
:cite:`ICML'03:zinkvich,NIPS'18:Zhang-Ader,JMLR:sword++,JMLR'21:BCO,journal'07:Hazan-adaptive,ICML'15:Daniely-adaptive,ICML19:Zhang-SACS`.
By providing a unified view to understand many algorithms proposed in the
literature, we argue that there are three critical algorithmic components:
**base-learner**, **meta-learner**, and **schedule**.
With such a perspective, we present systematic and modular Python
implementations for many online algorithms, packed in PyNOL. The package is
highly flexible and extensible, based on which one can define and implement her
own algorithms flexibly and conveniently. For example, we also implement some
classical algorithms for online MDPs based on this package
:cite:`NIPS'13:MDP-Neu,IJCAI'21:SSP-Rosenberg,COLT'21:SSP-minimax,ICML'22:mdp`.

Installation
------------

PyNOL is currently hosted on `PyPI <https://pypi.org/project/pynol/>`_. It
requires Python >= 3.8. You can simply install PyNOL from PyPI with the
following command:

.. code-block:: bash

    $ pip install pynol

To use PyNOL by source code, download this repository and run the following
command:

.. code-block:: bash

    $ python setup.py build
    $ python setup.py install

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/introduction.rst
   tutorials/structure.rst
   tutorials/algorithms.rst
   tutorials/examples.rst
   tutorials/cite.rst
   tutorials/references.rst

.. toctree::
   :maxdepth: 1
   :caption: API Docs

   apis/pynol.environment.rst
   apis/pynol.learner.rst
   apis/pynol.online_learning.rst
   apis/pynol.utils.rst

.. toctree::
   :maxdepth: 1
   :caption: Community

   contributors.rst

