pynol.learner
=============

base
''''

This module provides different base algorithms. On the one hand, they can be used to optimize static regret directly, on the other hand, they are crucial components for algorithms designed to optimize dynamic regret and adaptive regret.

Base
----

.. autoclass:: pynol.learner.base.Base
   :members:
   :undoc-members:
   :show-inheritance:

OGD
---

.. autoclass:: pynol.learner.base.OGD
   :members:
   :undoc-members:
   :show-inheritance:

BGDOnePoint
-----------

.. autoclass:: pynol.learner.base.BGDOnePoint
   :members:
   :undoc-members:
   :show-inheritance:

BGDTwoPoint
-----------

.. autoclass:: pynol.learner.base.BGDTwoPoint
   :members:
   :undoc-members:
   :show-inheritance:

SOGD
----

.. autoclass:: pynol.learner.base.SOGD
   :members:
   :undoc-members:
   :show-inheritance:


OptimisticBase
--------------

.. autoclass:: pynol.learner.base.OptimisticBase
   :members:
   :undoc-members:
   :show-inheritance:

OptimisticOGD
-------------

.. autoclass:: pynol.learner.base.OptimisticOGD
   :members:
   :undoc-members:
   :show-inheritance:

OEGD
----

.. autoclass:: pynol.learner.base.OEGD
   :members:
   :undoc-members:
   :show-inheritance:

Hedge
-----

.. autoclass:: pynol.learner.base.Hedge
   :members:
   :undoc-members:
   :show-inheritance:

OptimisticHedge
---------------

.. autoclass:: pynol.learner.base.OptimisticHedge
   :members:
   :undoc-members:
   :show-inheritance:

meta
''''

This module provides different meta algorithms. On the one hand, they can be
used to solve the expert-tracking problem directly, on the other hand, they are
crucial components to combine the base-learners to optimize dynamic regret and
adaptive regret.

Meta
----

.. autoclass:: pynol.learner.meta.Meta
   :members:
   :undoc-members:
   :show-inheritance:

Hedge
-----

.. autoclass:: pynol.learner.meta.Hedge
   :members:
   :undoc-members:
   :show-inheritance:

OptimisticMeta
--------------

.. autoclass:: pynol.learner.meta.OptimisticMeta
   :members:
   :undoc-members:
   :show-inheritance:

OptimisticLR
^^^^^^^^^^^^

.. autoclass:: pynol.learner.meta.OptimisticLR
   :members:
   :undoc-members:
   :show-inheritance:

OptimisticHedge
---------------

.. autoclass:: pynol.learner.meta.OptimisticHedge
   :members:
   :undoc-members:
   :show-inheritance:

MSMWC
-----

.. autoclass:: pynol.learner.meta.MSMWC
   :members:
   :undoc-members:
   :show-inheritance:

Prod
------

.. autoclass:: pynol.learner.meta.Prod
   :members:
   :show-inheritance:

AdaNormalHedge
--------------

.. autoclass:: pynol.learner.meta.AdaNormalHedge
   :members:
   :show-inheritance:

AFLHMeta
--------

.. autoclass:: pynol.learner.meta.AFLHMeta
   :members:
   :show-inheritance:

schedule
''''''''

``pynol.learner.schedule`` is the component to deal with the non-stationary
environments. This module consists of two main parts: ``SSP``
and ``Cover``. ``SSP`` specifies how to initialize the base-learners, which is
important for dynamic algorithms. The dynamic algorithms construct a step size
pool at first, and then initialize multiple base-learners, each employs a
specific step size. The construction is based on exponential discretization of
possible range of the optimal step size that usually depends on unknown
path-length. ``Cover`` contains different interval partitions (Cover) that base-learners
will last for, such as data stream cover (DSC), geometric cover (GC), compact GC
(CGC) and its problem-dependent version CPGC, which is important for adaptive
algorithm.

SSP
---

SSP
^^^

.. autoclass:: pynol.learner.schedule.ssp.SSP
   :members:
   :undoc-members:
   :show-inheritance:

StepSizeFreeSSP
^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.schedule.ssp.StepSizeFreeSSP
   :members:
   :undoc-members:
   :show-inheritance:

DiscreteSSP
^^^^^^^^^^^

.. autoclass:: pynol.learner.schedule.ssp.DiscreteSSP
   :members:
   :undoc-members:
   :show-inheritance:

Cover
-----

Cover
^^^^^

.. autoclass:: pynol.learner.schedule.cover.Cover
   :members:
   :undoc-members:
   :show-inheritance:

FullCover
^^^^^^^^^

.. autoclass:: pynol.learner.schedule.cover.FullCover
   :members:
   :undoc-members:
   :show-inheritance:

GC
^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.schedule.cover.GC
   :members:
   :undoc-members:
   :show-inheritance:

CGC
^^^^^^^^^

.. autoclass:: pynol.learner.schedule.cover.CGC
   :members:
   :undoc-members:
   :show-inheritance:

PCover
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.schedule.cover.PCover
   :members:
   :undoc-members:
   :show-inheritance:

PGC
^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.schedule.cover.PGC
   :members:
   :undoc-members:
   :show-inheritance:

PCGC
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.schedule.cover.PCGC
   :members:
   :undoc-members:
   :show-inheritance:

Schedule
--------

Schedule
^^^^^^^^

.. autoclass:: pynol.learner.schedule.schedule.Schedule
   :members:
   :undoc-members:
   :show-inheritance:

PSchedule
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.schedule.schedule.PSchedule
   :members:
   :undoc-members:
   :show-inheritance:

specification
'''''''''''''

Besides ``Base``, ``Meta`` and ``Schedule`` three main components, the remaining
parts of algorithms are collectively referred to ``specification``, which mainly
includes the design of ``optimism`` and ``surrogate loss``. As many algorithms
can be viewed as specials cases of Optimistic Mirror Descent, the construction
of ``optimism`` is crucial in algorithmic design. Moreover, replacing original
loss function by surrogate loss is a useful technique which can bring great
benefits sometimes.

SurrogateBase
-------------

SurrogateBase
^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.surrogate_base.SurrogateBase
   :members:
   :undoc-members:
   :show-inheritance:

LinearSurrogateBase
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.surrogate_base.LinearSurrogateBase
   :members:
   :undoc-members:
   :show-inheritance:

InnerSurrogateBase
^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.surrogate_base.InnerSurrogateBase
   :members:
   :undoc-members:
   :show-inheritance:

OptimismBase
-------------

OptimismBase
^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.optimism_base.OptimismBase
   :members:
   :undoc-members:
   :show-inheritance:

EnvironmentalOptimismBase
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.optimism_base.EnvironmentalOptimismBase
   :members:
   :undoc-members:
   :show-inheritance:

LastGradOptimismBase
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.optimism_base.LastGradOptimismBase
   :members:
   :undoc-members:
   :show-inheritance:

SurrogateMeta
-------------

SurrogateMeta
^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.surrogate_meta.SurrogateMeta
   :members:
   :undoc-members:
   :show-inheritance:

SurrogateMetaFromBase
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.surrogate_meta.SurrogateMetaFromBase
   :members:
   :undoc-members:
   :show-inheritance:

InnerSurrogateMeta
^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.surrogate_meta.InnerSurrogateMeta
   :members:
   :undoc-members:
   :show-inheritance:

InnerSwitchingSurrogateMeta
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.surrogate_meta.InnerSwitchingSurrogateMeta
   :members:
   :show-inheritance:

OptimismMeta
-------------

OptimismMeta
^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.optimism_meta.OptimismMeta
   :members:
   :undoc-members:
   :show-inheritance:

InnerSwitchingOptimismMeta
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.optimism_meta.InnerSwitchingOptimismMeta
   :members:
   :undoc-members:
   :show-inheritance:

InnerOptimismMeta
^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.optimism_meta.InnerOptimismMeta
   :members:
   :undoc-members:
   :show-inheritance:

SwordVariationOptimismMeta
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.optimism_meta.SwordVariationOptimismMeta
   :members:
   :undoc-members:
   :show-inheritance:

SwordBestOptimismMeta
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.optimism_meta.SwordBestOptimismMeta
   :members:
   :undoc-members:
   :show-inheritance:

Perturbation
-------------

Perturbation
^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.perturbation.Perturbation
   :members:
   :undoc-members:
   :show-inheritance:

OnePointPerturbation
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.perturbation.OnePointPerturbation
   :members:
   :undoc-members:
   :show-inheritance:

TwoPointPerturbation
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.specification.perturbation.TwoPointPerturbation
   :members:
   :undoc-members:
   :show-inheritance:

models
''''''

Model
-----

.. autoclass:: pynol.learner.models.model.Model
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic
-------

Ader
^^^^

.. autoclass:: pynol.learner.models.dynamic.ader.Ader
   :members:
   :undoc-members:
   :show-inheritance:

PBGDOnePoint
^^^^^^^^^^^^

.. autoclass:: pynol.learner.models.dynamic.pbgd.PBGDOnePoint
   :members:
   :undoc-members:
   :show-inheritance:

PBGDTwoPoint
^^^^^^^^^^^^

.. autoclass:: pynol.learner.models.dynamic.pbgd.PBGDTwoPoint
   :members:
   :undoc-members:
   :show-inheritance:

SwordVariation
^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.models.dynamic.sword.SwordVariation
   :members:
   :undoc-members:
   :show-inheritance:

SwordSmallLoss
^^^^^^^^^^^^^^

.. autoclass:: pynol.learner.models.dynamic.sword.SwordSmallLoss
   :members:
   :undoc-members:
   :show-inheritance:

SwordBest
^^^^^^^^^

.. autoclass:: pynol.learner.models.dynamic.sword.SwordBest
   :members:
   :undoc-members:
   :show-inheritance:

Sword++
^^^^^^^

.. autoclass:: pynol.learner.models.dynamic.swordpp.SwordPP
   :members:
   :undoc-members:
   :show-inheritance:

Scream
^^^^^^

.. autoclass:: pynol.learner.models.dynamic.scream.Scream
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive
--------

AFLH
^^^^

.. autoclass:: pynol.learner.models.adaptive.aflh.AFLH
   :members:
   :undoc-members:
   :show-inheritance:

SAOL
^^^^

.. autoclass:: pynol.learner.models.adaptive.saol.SAOL
   :members:
   :undoc-members:
   :show-inheritance:

SACS
^^^^

.. autoclass:: pynol.learner.models.adaptive.sacs.SACS
   :members:
   :undoc-members:
   :show-inheritance:

PSACS
^^^^^

.. autoclass:: pynol.learner.models.adaptive.sacs.PSACS
   :members:
   :undoc-members:
   :show-inheritance: