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