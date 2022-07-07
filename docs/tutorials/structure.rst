Structure
=========

Since OCO can be viewed as a repeat game between a learner and environments,
PyNOL follows this structure and thus consists of two parts logically:
``learner`` and ``environment``.

- ``learner``: this module defines the strategy employed by the online learner, and
  one can use our predefined models such as ``OGD``, ``Ader``, ``SAOL`` and so
  on, or define own model by flexibly combining the modules in ``base``,
  ``meta``, ``schedule`` and ``specification``, which will be introduced in
  detail later.
- ``environment``: this module consists of ``domain`` and ``loss function``. Before
  the game starts, the environment chooses the feasible set :math:`\mathcal{X}`
  by ``domain`` and passes to the ``learner``. And at each iteration :math:`t`,  the
  learner choose a decision :math:`x_t\in \mathcal{X}` in the ``domain`` and
  simultaneously the environment reveals loss function :math:`f_t` by ``loss
  function``.

 The overall structure of PyNOL looks like:

 .. image:: ../_static/figures/PyNOL_structure.png



 Firstly, we introduce the ``learner`` part in details. As mentioned earlier,
 we present a unified view to understand many existing learning algorithms for
 dynamic regret or adaptive regret minimization. As a result, the ``learner``
 component includes ``base``, ``meta``, ``schedule``, and ``specification``.

- ``base``: this module implements the base-learner, a particular online
  algorithm that can achieve low regret given a specific path-length (for
  dynamic regret) or a specific interval (for adaptive regret). We implement
  online gradient descent (OGD) :cite:`ICML'03:zinkvich`, bandit gradient descent (BGD) with
  one-point feedback :cite:`SODA'05:Flaxman-BCO` and two-point feedback
  :cite:`COLT'10:BCO-two-point`, online extra-gradient  descent (OEGD)
  :cite:`COLT'12:variation-Yang`, optimistic online gradient descent (Optimistic OGD)
  :cite:`COLT13:optimistic` and scale-free online gradient descent (SOGD)
  :cite:`TCS'18:SOGD`.
- ``meta``: this module implements the meta-learner, used to combine
  intermediate decisions from base learners. PyNOL includes Hedge
  :cite:`journals/iandc/LittlestoneW94`, Optimistic Hedge
  :cite:`COLT13:optimistic`,  MSMWC :cite:`COLT'21:impossible-tuning`, AFLHMeta
  :cite:`journal'07:Hazan-adaptive`, Prod
  :cite:`MLJ'17:Cesa-Prod,ICML'15:Daniely-adaptive` and AdaNormalHedge
  :cite:`COLT'15:Luo-AdaNormalHedge`.
- ``schedule``: this module consists of two parts: ``SSP`` and ``Cover``.
  ``SSP`` specifies how to initialize the base-learners, which is important for
  dynamic algorithms. The dynamic algorithms construct a step size pool (SSP) at
  first and then initialize multiple base-learners, each employs a specific
  step size. The construction is based on exponential discretization of possible
  range of the optimal step size that usually depends on unknown path-length.
  ``Cover`` contains different interval partitions (Cover) that base-learners
  will last for, such as geometric cover (GC)
  :cite:`ICML'09:Hazan-adaptive,TIT'12:andras,ICML'15:Daniely-adaptive`, compact
  GC (CGC) and its problem-dependent version PCGC :cite:`ICML19:Zhang-SACS`,
  which is important for adaptive algorithms.
- ``specification``: Besides these three main components, the remaining parts of
  algorithms are collectively referred to ``specification``, which mainly
  includes the design of ``optimism`` and ``surrogate loss``. As many algorithms
  can be viewed as specials cases of Optimistic Mirror Descent, the construction
  of ``optimism`` is crucial in algorithmic design. Moreover, replacing original
  loss by ``surrogate loss`` is a useful technique which can bring great
  benefits sometimes.

The overall structure is as follows. Online learner maintains a bunch of
base-learners according to some dedicated schedule and then employs the
meta-learner to ensemble them all to hedge the uncertainty. With this structure,
PyNOL eases the extension/modification of existing models, as well as the
creation of new models by the implemented APIs.

.. image:: ../_static/figures/structure.png

Next, we introduce ``environment``, which defines the environment by
``domain`` and ``loss function``:

- ``domain``: this module defines the feasible set of the learner's decision.
  In this module, we provide two common feasible sets: Euclidean ball and
  simplex. Users can define their desired type of feasible set in this module.
- ``loss function``: this module defines the loss function revealed by the
  environment. In this module, one can find common loss functions to use:
  ``logarithmic loss``, ``squared loss`` and so on. Similarly, users can define more loss
  functions easily without having to give the form of derivative function since
  it is computed automatically by ``autograd``.

In short, one can define a new model with ``base``, ``meta``, ``schedule``,
``specification`` and define the experiment environment with ``domain``, ``loss
function`` easily and quickly. Combining ``learner`` and ``environment``, user
completes the construction of the online learning procedure.