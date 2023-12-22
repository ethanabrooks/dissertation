#import "math.typ": *
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms
#import "@preview/tablex:0.0.7": tablex, rowspanx, colspanx

#show: show-algorithms

#set heading(numbering: "1.1")

= Algorithm Distillation + Policy Iteration

Generalization to novel tasks is an important challenge in multitask
reinforcement learning (RL). This setting entails a training regime that
includes a diversity of dynamics or reward functions, and a test regime that
evaluates the agent on unseen dynamics or reward functions. A typical approach
to this challenge is as follows: train a policy on the training tasks and use
various mechanisms to adapt the policy to the novel setting. In this work, we
explore the benefits of focusing instead on adapting a _model_ as a means to
adapt a policy, as opposed to adapting the policy directly. If the model
successfully adapts to the evaluation setting, we may use a variety of planning
methods to recover a good policy.

Our approach builds on the previous chapter, @chap:pi[Chapter], using a similar
methodology to choose actions on the downstream evaluation task. Concretely, at
each state, we use our learned model to perform multiple rollouts, each starting
from a different action in the action space. We use these rollouts to estimate
state-action values. Our behavior policy (as opposed to the policy used in the
rollouts) chooses the action corresponding to the highest estimate. As we
demonstrated in that work, this approach implements a form of policy iteration,
an algorithm proven to eventually converge to the optimal policy.

Our previous work used generic foundation models, namely Codex
#cites(<chen2021evaluating>), to implement both a world model and a policy
during the aforementioned rollouts. In contrast, our present work assumes access
to a dataset of RL trajectories and uses this data to train a causal transformer
#cites(<vaswani2017attention>) as a world model. Because the transformer is
trained on RL data and the evaluation task is similarly distributed to the
training tasks, the transformer is capable of modeling more complex domains than
the general-purpose foundation models from which we elicited world-model
predictions using a variety of prompting techniques. Another advantage of
training the model from scratch is that we can more easily assess generalization
by comparing training tasks with evaluation tasks. Such comparisons are not
possible when using a foundation model trained on a large, opaque dataset.

One possible rationale for our hypothesis, that planning with an adapted model
will generalize better than direct policy adaptation, is that a model relies
only on local information whereas a policy relies on non-local information.
However, model-based planning gains this advantage at the cost of compound
error---the tendency for any approximate model to accumulate error in an
auto-regressive chain, compounding the errors of new predictions with the errors
of prior predictions on which they are conditioned. Why should we expect the
benefit to outweigh the cost?

We argue that a function approximator can only learn a policy one of two ways:
either it learns the underlying logic of the policy, or it memorizes the policy
without distilling this logic---a strategy which does not generalize. However,
this underlying logic entails some kind of implicit value estimation, which
entails implicit modeling of the environment. In principle, these implicit
operations are susceptible to compound error accumulation in much the same way
as explicit model-based planning methods. Therefore the switch to model-based
planning does not actually introduce compound error as a new cost. Meanwhile,
model-based planning does eliminate the possibility of policy memorization,
since actions and values are not inferred directly but computed. Our conclusion
is that the net benefit of model-based planning for generalization is positive.

We tested this hypothesis in two sets of domains, a gridworld with randomly
spawned walls, and a simulated robotics domain implemented in Brax
#cites(<freeman2021brax>). In the gridworld setting, we found that the
model-based approach consistently outperformed a direct policy adaptation
approach, even as we varied the difficulty of generalization and varied the
quantity of training data. In the robotics domain, our results were
inconclusive, with the model-based approach outperforming the direct policy
adaptation approach on some domains, but not others.Generalization to novel
tasks is an important challenge in multitask reinforcement learning (RL). This
setting entails a training regime that includes a diversity of dynamics or
reward functions, and a test regime that evaluates the agent on unseen dynamics
or reward functions. A typical approach to this challenge is as follows: train a
policy on the training tasks and use various mechanisms to adapt the policy to
the novel setting. In this work, we explore the benefits of focusing instead on
adapting a _model_ as a means to adapt a policy, as opposed to adapting the
policy directly. If the model successfully adapts to the evaluation setting, we
may use a variety of planning methods to recover a good policy.

Our approach builds on the previous chapter, @chap:pi[Chapter], using a similar
methodology to choose actions on the downstream evaluation task. Concretely, at
each state, we use our learned model to perform multiple rollouts, each starting
from a different action in the action space. We use these rollouts to estimate
state-action values. Our behavior policy (as opposed to the policy used in the
rollouts) chooses the action corresponding to the highest estimate. As we
demonstrated in that work, this approach implements a form of policy iteration,
an algorithm proven to eventually converge to the optimal policy.

Our previous work used generic foundation models, namely Codex
#cites(<chen2021evaluating>), to implement both a world model and a policy
during the aforementioned rollouts. In contrast, our present work assumes access
to a dataset of RL trajectories and uses this data to train a causal transformer
#cites(<vaswani2017attention>) as a world model. Because the transformer is
trained on RL data and the evaluation task is similarly distributed to the
training tasks, the transformer is capable of modeling more complex domains than
the general-purpose foundation models from which we elicited world-model
predictions using a variety of prompting techniques. Another advantage of
training the model from scratch is that we can more easily assess generalization
by comparing training tasks with evaluation tasks. Such comparisons are not
possible when using a foundation model trained on a large, opaque dataset.

One possible rationale for our hypothesis, that planning with an adapted model
will generalize better than direct policy adaptation, is that a model relies
only on local information whereas a policy relies on non-local information.
However, model-based planning gains this advantage at the cost of compound error
---the tendency for any approximate model to accumulate error in an
auto-regressive chain, compounding the errors of new predictions with the errors
of prior predictions on which they are conditioned. Why should we expect the
benefit to outweigh the cost?

We argue that a function approximator can only learn a policy one of two ways:
either it learns the underlying logic of the policy, or it memorizes the policy
without distilling this logic---a strategy which does not generalize. However,
this underlying logic entails some kind of implicit value estimation, which
entails implicit modeling of the environment. In principle, these implicit
operations are susceptible to compound error accumulation in much the same way
as explicit model-based planning methods. Therefore the switch to model-based
planning does not actually introduce compound error as a new cost. Meanwhile,
model-based planning does eliminate the possibility of policy memorization,
since actions and values are not inferred directly but computed. Our conclusion
is that the net benefit of model-based planning for generalization is positive.

We tested this hypothesis in two sets of domains, a gridworld with randomly
spawned walls, and a simulated robotics domain implemented in Brax
#cites(<freeman2021brax>). In the gridworld setting, we found that the
model-based approach consistently outperformed a direct policy adaptation
approach, even as we varied the difficulty of generalization and varied the
quantity of training data. In the robotics domain, our results were
inconclusive, with the model-based approach outperforming the direct policy
adaptation approach on some domains, but not others.

== Background
*Markov Decision Processes*

= Dummy <chap:pi>
#bibliography("main.bib", style: "association-for-computing-machinery")