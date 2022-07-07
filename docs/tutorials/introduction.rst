Introduction
============

Online learning is studied in a variety of disciplines including optimization,
game theory, and information theory :cite:`cesa2006prediction`, which attracts
an increasing interest since it serves as the foundation of sequential
prediction and decision making problems :cite:`book'16:Hazan-OCO,book:Sutton-RL`. In
particular, online convex optimization (OCO) is a powerful framework to model
such sequential learning tasks. At time :math:`t`, the learner chooses a
decision :math:`x_t` from a convex set :math:`\mathcal{X}` and environments
reveal a convex function :math:`f_t: \mathcal{X} \mapsto \mathbb{R}`. The
learner then suffers an instantaneous loss :math:`f_t(x_t)`. The performance is
measured by (static) *regret*, defined as

.. math::

    \mbox{S-Regret}_T = \sum_{t=1}^T f_t(x_t) - \min_{x \in \mathcal{X}} \sum_{t=1}^T f_t(x),

which is the difference between cumulative loss suffered by the player and that
of the best fixed strategy. A low-regret algorithm yields a decision sequence
competitive to the best fixed decision, thus ensuring reasonably good
performance. However, when the environments are *non-stationary*, even the best
fixed strategy could still perform poorly, algorithms that optimize the static
regret are not guaranteed to perform well over the horizon.

To address this limitation, two different measures, dynamic regret and adaptive
regret, are proposed in the literature. Dynamic regret measures the online
learner's performance by competing with time-varying comparators while adaptive
regret examines the online learner's performance within any contiguous interval.
Specifically, since the best decision can change arbitrarily in the
non-stationary environments, dynamic regret :cite:`ICML'03:zinkvich` is proposed
to compete with *any* comparator sequence :math:`u_1, \ldots, u_T \in
\mathcal{X}`,

.. math::

    \mbox{D-Regret}_T(u_1, \ldots, u_T) = \sum_{t=1}^T f_t(x_t) -  \sum_{t=1}^T f_t(u_t).

The dynamic regret upper bound usually involves the dependence on the
path-length :math:`P_T = \sum_{t=2}^T \lVert u_t - u_{t-1}\rVert_2`, which
captures the amount of environmental non-stationarity. On the other hand,
instead of competing with the changing comparators, adaptive regret
:cite:`ICML'09:Hazan-adaptive,ICML'15:Daniely-adaptive` examines the local
performance and is defined as

.. math::

  \mbox{A-Regret}_T(\tau) = \max_{[s, s+\tau-1]\subseteq [T]} \left\{
  \sum_{t=s}^{s+\tau-1} f_t(x_t) - \min_{x \in \mathcal{X}}
  \sum_{t=s}^{s+\tau-1} f_t(x) \right\},

which aims to minimize the regret over *any* interval with length :math:`\tau``
such that the algorithm can track environment changes. The two measures draw
considerable attention recently,
see :cite:`conf/nips/Cesa-BianchiGLS12,TIT'12:andras,AISTATS'15:dynamic-optimistic,
NIPS'18:Zhang-Ader,UAI'20:simple, NIPS'20:sword,JMLR'21:BCO,ICML'22:mdp` for
dynamic regret and :cite:`ICML'09:Hazan-adaptive,ICML'15:Daniely-adaptive,
AISTATS'17:coin-betting-adaptive,ICML19:Zhang-SACS,NIPS'21:dual-adaptive` for
adaptive regret.

To optimize the two performance metrics (dynamic regret and adaptive regret),
various online algorithms are proposed in the past decades. We unify them from
the viewpoint of *online ensemble* :cite:`book'12:ensemble-zhou,thesis:zhao2021-eng`, which includes
three crucial components: **base-learner**, **meta-learner**, and **schedule**.
More specifically, the fundamental challenge of non-stationary online learning
is to handle the uncertainty (unknown non-stationarity) of the environments.
Thus it is kind of necessary to employ the meta-base structure, in which the
online learner maintains a bunch of base-learners according to some dedicated
schedule and then employs the meta-learner to ensemble them all to hedge the
uncertainty. With such a perspective, we present systematic and modular Python
implementations for those online algorithms, packed in PyNOL, a **Py**\thon
package for  **N**\on-stationary **O**\nline **L**\earning. PyNOL
package is highly flexible and extensible, based on which researchers can define
and implement their own algorithms flexibly and conveniently.
