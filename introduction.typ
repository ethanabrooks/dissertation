= Introduction <sec:introduction>

== The Importance of Rapid Adaptation
Deep neural networks optimized by gradient descent have defined the most
pioneering techniques of the last decade of machine learning research. While
other architectures may provide stronger stronger guarantees, they often require
certain assumptions such as linearity, convexity, and smoothness that are not
present in reality. As a result, researchers have turned to general-purpose
algorithms that make few assumptions about the functions they approximate.

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
_inner-loop_ optimizes performance on a specific task and an _outer-loop_
optimizes the learning procedure of the inner-loop. Some approaches use gradient
updates for both levels #cite(<finn2017model>)#cite(<stadie2018some>), while
others use gradients only for the outer-loop and train some kind of learned
update rule for the inner-loop.

== In-Context Learning
A common approach to learning an update rule involves learning some
representation of each new experience of the task, along with the parameters of
some operator that aggregates these representations. For example, the $"RL"^2$
#cite(<duan_rl2_2016>) algorithm uses the Long-Short Term Memory (LSTM,
#cite(<hochreiter1997long>)) architecture for aggregation. Others
#cite(<team2023human>) have used Transformers #cite(<vaswani2017attention>)
in place of LSTMs. PEARL #cite(<rakelly2019efficient>) uses a latent context
that accumulates representations of the agent's observations in a product of
Gaussian factors.

We call this aggregated representation _memory_ or _context_. In many settings,
one observes a rapid increase in performance as new experiences accumulate in
the context of a neural architecture. This increase in performance is what we
call _in-context learning_.

In the context of meta-learning, in-context learning corresponds to the
inner-loop, which rapidly adapts the learner to a specific task. However, not
all in-context learning uses the inner/outer-loop meta-learning formulation.
Famously, GPT-3 #cite(<brown2020language>)
demonstrated that a large language model <LLM> developed the ability to learn
in-context as an emergent consequence of large-scale training. Another
interesting example of in-context learning outside of meta-learning is Algorithm
Distillation #cite(<laskin2022context>), which demonstrates that a transformer
trained on offline RL data can distill and reproduce improvement operators from
the algorithm that generated the data.

== Meta Reinforcement Learning
Meta reinforcement learning (meta-RL) is a subdiscipline of meta-learning which
focuses on reinforcement learning (RL) tasks. Meta-RL typically targets
multi-task settings in which the agent cannot infer the current task from any
single observation, but must discover its properties through exploration.
Existing literature #cite(<finn2017model>)#cite(<rakelly2019efficient>)#cite(<zintgraf2019varibad>) typically
evaluates meta-RL agents in special multi-trial episodes in which the agent's
state resets once per trial and multiple times per episode, but the task remains
the same. The agent must maximize cumulative or final reward per episode, with
optimal performance requiring the agent to exploit in later trials information
that was discovered in earlier trials.

This setting demands a different strategy than multi-task RL, in which a fully
trained agent exploits a near-optimal policy, even in held-out evaluation
settings. A meta-RL agent must learn to explore initially and later transition
to exploitation. For example, in a gridworld, efficient exploration may entail a
circuitous search policy that never revisits any state more than once. Meanwhile
an exploitation policy moves in a straight path toward some goal, which earlier
exploration has revealed.

== Meta-Learning Challenges
The outer/inner-loop meta-learning formulation is powerful and general. However,
the approach has difficulty scaling to complex problems with large task spaces.
One of the difficulties inherent in this form of meta-learning is the coupling
of the inner- and outer-loop optimizations. This causes instability in the
learning process and forces the meta-learning algorithm to search not only the
space of task solutions, but also the space of learning algorithms that might
produce each solution. When learning signal is sparse or the task space is
large, the outer loop will often fail to make progress.

Meta-RL algorithms are also susceptible to a local minimum in which they fail to
recognize that the information yielded from exploration can inform an efficient
exploitation policy, remaining in a permanent suboptimal exploration mode. In
our gridworld setting, the agent might fail to recognize that the goal appears
in the same location in every trial of an episode, repeatedly searching for it
instead of moving toward it efficiently. AdA #cite(<team2023human>) is one of
the only meta-RL works to have scaled this approach to complex domains, but only
with the help of sophisticated curriculum techniques which may not apply to all
settings.

== An Alternative Approach to In-Context Learning
This thesis explores a meta-RL in-context-learning methodology which decouples
the inner- and outer-loop by sending learning signal in one direction only, from
the meta-process to the inner process, and from an _external dataset_ to the
meta-process. Without feedback from the inner process, the meta-process is
unable to completely specialize to the downstream, inner-process setting, which
limits its ability to match traditional meta-learning in terms of downstream
efficiency, but yields other benefits. In particular, we note that our methods
can leverage large pre-existing offline datasets but unlike existing offline RL
techniques #cite(<fujimoto2019off>)#cite(<kumar2020conservative>), they permit
further interaction with and exploration of the environment. We defer the exact
details of our methods to later chapters, but offer a cursory sketch at this
point.

=== Value Estimation
We use a model trained on offline data to make context-conditional predictions,
that we use to estimate state-action values. This stage of training involves
standard gradient-based optimization and is analogous to outer-loop optimization
in a traditional meta-learning algorithm, insofar as this stage distills priors
in the model that support rapid downstream learning. The inputs to the model
contain information relating to the environment dynamics, the reward function,
and the current policy --- the parameters of a value estimate --- and we train
the model to condition its predictions on this information. We explore various
techniques to encourage the model to attend to this context (in-context
learning) as opposed to resorting to priors encoded in its weights (in-weights
learning). #cite(<chan2022data>, form: "prose") provides further discussion of
this distinction. The model will then demonstrate some capability to generalize
its predictions to unseen downstream settings, as long as those settings are
adequately represented in the inputs and these inputs are similarly distributed
to the inputs in the training dataset.

=== Policy Improvement
We target a meta-RL setting in which the agent is unable to perform optimally at
the start of an episode, due to limited knowledge of the task and environment.
Our method therefore requires some mechanism for improving the policy as
information accumulates through interaction. This stage of training does not use
gradients and is analogous to the inner-loop of a traditional meta-learning
algorithm, insofar as it adapts the learner to a specific task. To induce policy
improvement, our method combines the classic policy iteration algorithm with
in-context learning techniques. At each timestep, we use the method described in
the previous paragraph to estimate the state-action value of each action in the
action space. We then choose the action with the highest value estimate.

In general, policy iteration depends on a cycle in which actions are chosen
greedily with respect to value estimates and value estimates are updated to
track the newly improved policy. Our action selection method satisfies the first
half of this formulation. To satisfy the second, we condition our model's
predictions exclusively on data drawn from recent interactions with the
environment, which reflect or approximate the current policy. As the policy
improves, the behavioral data will capture this improvement, the model's
predictions will reflect the improvement through this data, and the greedy
action selection strategy will produce an improvement cycle.

== Summary of Chapters
=== In-Context Policy Iteration
Our first chapter explores a method that bootstraps a pre-existing large
language model --- Codex #cite(<chen2021evaluating>) --- to produce value
estimates. This work restricts itself to a set of toy domains that can be
translated into text. Instead of predicting value directly, a form of
quantitative reasoning that would strain the capabilities of a language model,
our method uses simulated rollouts to generate monte-carlo estimations of value,
similar to Model-Predictive Control #cite(<pan2022model>). The work presents a
series of prompting techniques for eliciting these predictions from a
pre-trained language model.

Continuing our analogy to traditional meta-learning, the outer-loop corresponds
to the pretraining of Codex. Of the three methods that we present in this work,
this is the purest instantiation of the "one-way learning signal" formulation
that we posited earlier, since the dataset on which Codex was trained was
completely agnostic to the downstream tasks on which we evaluate our method.
Nevertheless, we present evidence that our method leverages commonsense pattern
completion and numerical reasoning distilled in Codex's weights.

=== Algorithm Distillaion + Model-Based Planning
Our second chapter extends the first by training a new model from offline
reinforcement learning (RL) data instead of using an existing pre-trained
language model. We demonstrate that learning planning primitives from the
offline data helps speed up training and facilitate generalization to novel
dynamics and reward functions. Additionally, we demonstrate that in-context
policy iteration can be used in conjunction with Algorithm Distillation #cite(<laskin2022context>),
superimposing the policy improvement operators induced by both methods.

=== Bellman Update Networks
Our final chapter, in which we propose future work, describes an alternative to
the monte-carlo rollout estimation technique used in the other two chapters.
This chapter advocates a method which directly optimizes the accuracy of value
predictions, instead of optimizing surrogate world-model predictions. We
initially present a naive method for estimating values with a single inference
step, but contrast it with an iterative approach that learns to approximate
individual Bellman updates and generates estimates through repeated applications
of the model.