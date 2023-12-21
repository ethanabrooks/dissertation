= Abstract
In-Context Learning describes a form of learning that occurs when a model
accumulates information in its context or memory. For example, an LSTM can
rapidly adapt to a novel task as input/target exemplars are fed into it. While
in-context learning of this kind is not a new discovery, recent work has
demonstrated the capacity of large "foundation" models to acquire this ability
"for free,"" by training on large quantities of semi-supervised data, without
the sophisticated (but often unstable) meta-objectives proposed by many earlier
papers #cite(<finn2017model>)#cite(<stadie2018some>)#cite(<rakelly2019efficient>).
In this work we explore several algorithms which specialize in-context learning
based on semi-supervised methods to the reinforcement learning (RL) setting. In
particular, we explore three approaches to in-context learning of value
(expected cumulative discounted reward).

The first of these methods demonstrates a method for implementing policy
iteration, a classic RL algorithm, using a pre-trained large language model
(LLM). We use the LLM to generate planning rollouts and extract monte-carlo
estimates of value from them. We demonstrate the method on several small,
text-based domains and present evidence that the LLM can generalize to unseen
states, a key requirement of learning in non-tabular settings.

The second method imports many of the ideas of the first, but trains a
transformer model directly on offline RL data. We incorporate Algorithm
Distillation (AD) #cite(<laskin2022context>), another method for in-context
reinforcement learning that directly distills the improvement operator from data
that includes behavior ranging from random to optimal. Our method combines the
benefits of AD with the policy iteration method proposed in our previous work
and demonstrates benefits in performance and generalization.

Our third method proposes a new method for estimating value. Like the previous
methods, this one implements a form of policy iteration, but eschews monte-carlo
rollouts for a new approach to estimating value. We train a network to estimate
Bellman updates and iteratively feed its outputs back into itself until the
estimate converges. We find that this iterative approach improves the capability
of the value estimates to generalize and mitigates some of the instability of
other offline methods.