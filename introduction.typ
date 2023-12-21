= Introduction

== The Importance of Rapid Adaptation
Deep neural networks optimized by gradient descent have shown strong empirical
results in many complex and realistic settings. While other algorithms with
faster convergence and stronger guarantees exist, they often require certain
properties such as linearity, convexity, and smoothness that are not present in
reality. As a result, researchers have turned to general-purpose algorithms that
make few assumptions about the functions they approximate.

However, these algorithms' generality comes at a cost: they learn slowly and
require thousands or millions of gradient updates to converge. This contrasts
with human learners, who can quickly master new tasks after brief exposure.
Rapid adaptation is a key feature of human intelligence, allowing us to behave
intelligently and develop new skills in unfamiliar settings #cite(<lake2017building>)

In principle, we can retrain a deep network for each new task it encounters.
However, this approach has several drawbacks. Retraining requires long
interruptions and specialized datasets or simulations, requirements which many
realistic settings do not support. Additionally, we may not be able to
distinguish or define new tasks, making it difficult to determine when to
retrain and what task to retrain on. Finally, non-stationary approaches to
training neural networks are prone to various failure modes
#cite(<french1999catastrophic>)#cite(<kirkpatrick2017overcoming>)

== Meta-Learning
These concerns have reinvigorated a subdiscipline called "meta-learning"
#cite(<schmidhuber1987evolutionary>) within the deep learning community.
Meta-learning produces learning algorithms which acquire speed through
specialization to a setting, similar to earlier optimization algorithms. Unlike
these algorithms which commit to a priori assumptions, meta-learning discovers
the properties of its setting through trial-and-error interaction. It then uses
these properties to generate fast, on-the-fly learning algorithms.

In general, meta-learning algorithms have a hierarchical structure in which an
*inner-loop* optimizes performance on a specific task and an *outer-loop*
optimizes the learning procedure of the inner-loop. Some approaches use gradient
updates for both levels, #cite(<finn2017model>)#cite(<stadie2018some>) while
others others use gradients only for the outer-loop and train some kind of
learned update rule for the inner-loop.

== In-Context Learning
A common approach to learning an update rule involves learning some latent
embedding of each new experience of the task, along with the parameters of some
operator that aggregates these embeddings. For example, the// $ RL^2 $
#cite(<duan_rl2_2016>) algorithm uses the Long-Short Term Memory (LSTM,
#cite(<hochreiter1997long>)) architecture for aggregation. Others
#cite(<team2023human>) have used Transformers #cite(<vaswani2017attention>)
in place of LSTMs. PEARL #cite(<rakelly2019efficient>) uses a latent context
that accumulates representations of the agent's observations in a product of
Gaussian factors.