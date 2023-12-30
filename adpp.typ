#import "math.typ": *
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms
#import "@preview/tablex:0.0.7": tablex, rowspanx, colspanx

#show: show-algorithms

#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

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
<background>
==== Markov Decision Processes
<markov-decision-processes>
A Markov Decision Process \(MDP) is a problem formulation in which an agent
interacts with an environment through actions, receiving rewards for each
interaction. Each action transitions the environment from some state to some
other state and the agent is provided observations which depend on this state
\(and from which the state can usually be inferred). MDPs can be "fully" or "partially"
observed. A fully observed MDP is defined by the property that the distribution
for each transition and reward is fully conditioned on the current observation.
In a partially observed MDP, the distribution depends on history — some
observations preceding the current one. The goal of reinforcement learning is to
discover a "policy" — a mapping from observations to actions — which maximizes
cumulative reward over time.

==== Model-Based Planning
<model-based-planning>
A model is a function which approximates the transition and reward distributions
of the environment. Typically a model is not given but must be learned from
experience gathered through interaction with the environment. Model-Based
Planning describes a class of approaches to reinforcement learning which use a
model to choose actions. These approaches vary widely, but most take advantage
of the fact that an accurate model enables the agent to anticipate the
consequences of an action before taking it in the environment.

==== Transformers
<transformers>
Transformers #cite(<vaswani2017attention>) are a class of neural networks which
map an input sequence to an output sequence of the same length and utilize a
mechanism known as "self-attention." Self-attention maps the \$i^\\Th\$ element
of the input sequence to a key vector $k_i$, a value vector $v_i$, and query
vector $q_i$. The output of self-attention is a weighted sum of the value
vectors:
$ upright("Attention") lr((q_i comma k comma v)) eq sum_j upright("softmax") lr((q_i k_j slash sqrt(D))) v_j $
where $D$ is the dimensionality of the vectors. The softmax is applied accross
the inputs so that
$sum_j upright("softmax") lr((q_i k_j slash sqrt(D))) eq 1$. A typical
transformer applies this self-attention operation multiple times, interspersing
linear projections and layer-norm operations between each application. A #emph[causal] transformer
applies a mask to the attention operation which prevents the model from
attending from the \$i^\\Th\$ element to the \$j^\\Th\$ element if $i lt j$,
that is, if the \$j^\\Th\$ element appears later in the sequence.

==== Policy Iteration
<policy-iteration>
Policy iteration is a technique for improving a policy in which one estimates
the values for the current policy and then chooses actions greedily with respect
to these value estimates. This yields a new policy which is at least as good as
the original policy according to the policy improvement theorem \(assuming that
the value estimates are correct). This process may be repeated indefinitely
until convergence. To choose actions greedily with respect to the value
estimates, there must be some mechanism for estimating value conditioned not
only on the current state and policy but also on an arbitrary action. The "greedy"
action choice corresponds to the action with the highest value estimate. Policy
iteration is possible if our value estimates are unbiased for any state-action
pair and for the current policy.

==== Algorithm Distillation
<algorithm-distillation>
Algorithm Distillation #cite(<laskin2022context>) is a method for distilling the
logic of a source RL algorithm into a causal transformer. The method assumes
access to a dataset of "learning histories" generated by the source algorithm,
which comprise state, action, reward sequences spanning the entire course of
learning, starting with initial random behavior and ending with fully optimized
behavior. The transformer is trained to predict actions given the entire history
of behavior or some large subset of it. Optimal prediction of these actions
requires the model to capture not only the current policy but also the
improvements to that policy by the source algorithm. Auto-regressive rollouts of
the distilled model, in which actions predicted by the model are actually used
to interact with the environment and then cycled back into the model input,
demonstrate improvement of the policy without any adjustment to the model's
parameters, indicating that the model does, in fact, capture some policy
improvement logic from the source algorithm.

== Method
<method>
In this section, we describe the details of our proposed algorithm, which we
call #ADPP-full-name (#ADPP). We assume a dataset of $N$ trajectories generated
by interactions with an environment:

$ Dataset := (
(Obs^n_0, Act^n_0, Rew^n_0, Ter^n_0, dots, Obs^n_T, Act^n_T,
Rew^n_T, Ter^n_T) sim Policy_n )_(n=1)^N $

with $Obs^n_t$ referring to the $t^"th"$ observation in the $n^"th"$
trajectory, $Act^n_t$ to the $t^"th"$ action, $Rew^n_t$ to the
$t^"th"$ reward, and $Ter[t](n)$ to the $t^"th"$ termination boolean. Each
trajectory corresponds with a single task $Task$ and a single policy
$Policy$, but the dataset contains as many as $N$ tasks and policies. We also
introduce the following nomenclature for trajectory histories:

$ History^n_t := (Act^n_(t-Recency),
Rew^n_(t-Recency), Ter^n_(t-Recency), Obs^n_(t-Recency+1), dots,
Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t) $ <eq:history>

where $Recency$ is a fixed hyperparameter.

== Model Training
<model-training>
Given such a dataset, we may train a world-model with a negative log likelihood
\(NLL) loss:

$ Loss_theta := -sum_(n=1)^N sum_(t=1)^(T-1) log Prob_theta (Act^n_t |
History^n_t) + log Prob_theta (Rew^n_t, Ter^n_t, Obs^n_(t+1) | History^n_t,
Act^n_t) $

In this work we implement $Prob_theta$ as a causal transformer. For
action-prediction $Prob_theta (Act^n_t |History^n_t) $, the inputs comprise
chronological
$(Act^n_(i-1) Rew^n_(i-1) Ter^n_(i-1) Obs^n_i)_(i=t-Recency)^t$
tuples. Each index of the transformer comprises one such tuple, with each
component of the tuple embedded and the embeddings concatenated. For the other
predictions $Prob_theta (Rew^n_t, Ter^n_t, Obs^n_(t+1) | History^n_t, Act^n_t)$,
we use the same procedure but rotate each tuple such that index $i$
corresponds to
$Rew^n_(i-1) Ter^n_(i-1) Obs^n_(i-1) Act^n_i$.

== Downstream Evaluation
<downstream-evaluation>
#algorithm-figure({
  import "algorithmic.typ": *
  algorithm(..Function(
    $QValue(History_t, Act)$,
    State($u gets 1$, comment: "time step for rollout"),
    State($Act^u gets Act$),
    ..While("termination condition not met", State(
      $Rew^u, Ter^u, Obs^(u+1) sim Prob_theta (dot.c|History_t, Act^u)$,
      comment: "Model reward, termination, next observation",
    ), State(
      $a^(u+1) sim Prob_theta (dot.c|History_t, Rew_t, Ter_t, Obs_(t+1))$,
      comment: "Model action",
    ), State($u gets u + 1$, comment: "Append predictions to history.")),
    Return($sum_(u=0)^t gamma^(u-1) Rew_u$),
  ))
}, caption: [ Estimating Q-values with monte-carlo rollouts. ])

Our approach to choosing actions during the downstream evaluation is as follows.
For each action $Act$ in a set of candidate actions (either our complete action
space or some subset thereof), we compute a state-action value estimate
$QValue(History_t, Act)$, where $History_t$ is the history defined in
@eq:history. We do this by modeling a rollout from the current state
(conditioned on the history $History_t$) and from $Act$. Modelling the rollout
is a cycle of sampling values from the model and feeding them back into the
model auto-regressively. First we sample
$(Rew_t, Ter_t, Obs_(t+1))$. Based on this prediction, we sample
$Act_(t+1)$. Adding this to the input allows us to sample
$(Rew_(t+1), Ter_(t+1), Obs_(t+2))$, repeating this cycle until some termination
condition is reached. For example, we might terminate the process once
$Ter_i$ is true, or once the rollout reaches some maximum length.

The final step is to choose an action. Having modelled the rollout, we compute
the value estimate as the discounted sum of rewards in the rollout. See
Algorithm for a pseudocode implementation. Repeating this process of modelling
rollouts and computing value estimates for each action in pool of candidate
actions, we simply choose the action with the highest value estimate:
$arg max_(Act in Actions) QValue(History_t, Act)$.

== Policy improvement
<policy-improvement>
If the downstream task is sufficiently dissimilar to the tasks in the training
dataset, or if the training dataset does not contain optimal policies, then it
is unlikely that the procedure described above will initially yield an optimal
policy. Some method for policy improvement will be necessary.

==== Policy Iteration
Our method satisfies this requirement by implementing a form a policy iteration.
To see this, first observe that our model is always trained to map a behavior
history drawn from a single policy to actions drawn from the same policy. A
fully trained model will therefore learn to match the distribution of actions in
its output to the distribution of actions in its input. Since our rollouts are
conditioned on histories drawn from our behavior policy, the rollout policy will
approximately match this policy. Our value estimates will therefore be
conditioned on the current behavior policy. However, by choosing the action
corresponding to $arg max_(Act in Actions)
QValue(History_t, Act)$, our behavior policy always improves on the policy on
which $QValue(History_t, Act)$ is conditioned, a consequence of the policy
improvement theorem. Thus, each time an action is chosen using this $arg max$ method,
our behavior policy improves on itself.

Walking through this process step by step, suppose $Policy^n$ is some policy
that we use to behave. We collect a trajectory containing actions drawn from
this policy. When we perform rollouts, we condition on this trajectory and the
rollout policy simulates $Policy^n$. Assuming that this simulation is accurate
as well as the world model, our value estimate will be an unbiased monte carlo
estimate of $QValue(History_t, Act)$ for any action $Act$. Then we act with
policy $Policy^(n+1) := arg max_(Act in Actions)
QValue(History_t, Act)$. But $Policy^(n+1)$ is at least as good as $Policy^n$.
Using the same reasoning, $Policy^(n+2)$ will be at least as good as $Policy^(n+1)$,
and so on. Note that in our implementation, we perform the $arg max$ at each
step, rather than first collecting a full trajectory with a single policy.

==== Algorithm Distillation
Our setting is almost identical to Algorithm Distillation (AD), if we include
the assumption that trajectories include a full learning history. Rather than
competing with AD, our method is actually complementary to it. If the input to
our transformer is a sufficiently long history of behavior, then the rollout
policy will not only match the input policy but actually improve upon it, as
demonstrated in that paper. Then $QValue$ will actually estimate values for a
policy $Policy'_n$ that is at least as good as the input policy
$Policy_n$. Then $V^(Policy_(n+1)) >= V^(Policy'_n) >=
V^(Policy_n)$ Therefore each step of improvement actually superimposes the two
improvement operators, one from the $arg max$ operator, the other from AD.

== Experiments
<experiments>
==== Domains
<domains>
In this work, we chose to focus on partially observable domains. This ensures
that both the initial policy and the initial model in our downstream domain will
be suboptimal, since the true dynamics or reward function cannot be inferred
until the agent has gathered experience. Recovering the optimal policy will
require policy improvement along side model improvement. Model improvement will
occur as the agent collects experience and the transformer context is populated
with transitions drawn from the current dynamics and reward functions. Policy
improvement relies on the mechanisms detailed in the previous sections. One
hypothesis that our experiments test is whether these learning processes can
successfully happen concurrently.

All of our experiments occur within a discrete, partially observable,
$5 times 5$ grid world. The agent has four actions, up, left, down, right. For
each task, we spawn a key and a door in a random location. The agent receives a
reward of 1 for visiting the key location and then a reward of 1 for visiting
the door. The agent observes only its own position and must infer the positions
of the key and the door from the history of interactions which the transformer
prompt contains. The episode terminates when the agent visits the door, or after
25 time-steps.

==== Baselines
<baselines>
Currently we compare our method with two baselines: vanilla AD and
"Ground-Truth" ICMBP. The latter is identical to our method, but uses a
ground-truth model of the environment, which is free of error. The comparison
with AD highlights the contribution of the model-based planning component of our
method. The comparison with Ground-Truth ICMBP establishes an approximate upper
bound for our method and distinguishes the contribution of model error to the RL
metrics that we record.

== Results
<results>

#figure(image("figures/adpp/no-walls.png"), caption: [
  Evaluation on withheld location pairs.
])
#grid(columns: (auto, auto), {
  show figure: it => [
    #align(center)[#it.body]
    #set align(left)
    #pad(x: .5cm)[#it.caption ]
  ]
  figure(
    image("figures/adpp/model-accuracy.png", height: 100pt),
    caption: [dummy],
  )
}, {
  show figure: it => [
    #align(center)[#it.body]
    #set align(left)
    #pad(x: 1.1cm)[#it.caption ]
  ]
  figure(
    image("figures/adpp/unseen-goals.png", height: 100pt),
    caption: [ Evaluation on fully withheld locations. ],
  )
})

==== Evaluation on Withheld Goals
<evaluation-on-withheld-goals>
In our first experiment, we evaluate the agent on a set of withheld key-door
pairs, which we sample uniformly at random (10% of all possible pairs) and
remove from the training set. As figure indicates, our algorithm outperforms the
AD baseline both in time to converge and final performance. We attribute this to
the fact that our method's downstream policy directly optimize expected return,
choosing actions that correspond to the highest value estimate. In contrast,
AD's policy only maximizes return by proxy — maximizing the probability of the
actions of a source algorithm which in turn maximizes expected return. This
indirection contributes noise to the downstream policy through modeling error.
Moreover, we note that our method completely recovers the performance of the
ground-truth baseline, though its speed of convergence lags behind slightly, due
to the initial exploration phase in which the model learns the reward function
through trial and error.

Next, we increase the generalization challenge by holding out key and door
locations entirely, never training the agent source algorithm on keys or doors
in the upper-left four cells of the grid and then placing both keys and doors
exclusively within this region during downstream evaluation. As figure
demonstrates, AD generalizes poorly in this setting, on average discovering only
one of the two goals. In contrast, our method maintains relatively high
performance. We attribute this to the fact that our method learns low-level
planning primitives (the reward function), which generalize better than
high-level abstractions like a policy. As we argued in section , higher-level
abstractions are prone to memorization since they do not always distill the
logic which produced them.

#figure(
  [#box(image("figures/adpp/generalization-to-more-walls-timestep.png"))],
  caption: [
    Generalization to higher percentages of walls.
  ],
)

==== Evaluation on Withheld Wall Configurations
<evaluation-on-withheld-wall-configurations>
In addition to evaluating generalization to novel reward functions, we also
evaluated our method's ability to generalize to novel dynamics. We did this by
adding walls to the grid world, which obstruct the agent's movement. During
training we placed the walls at all possible locations, sampled IID, with 10%
probability. During evaluation, we tested the agent on equal or higher
percentages of wall placement. As indicated by figure , out method maintains
performance and nearly matches the ground-truth version, while AD's performance
rapidly degrades. Again we attribute this to the tendency of lower-level
primitives to generalize better than higher-level abstractions.

#figure(
  [#box(image("figures/adpp/more-walls-achievable-timestep.png"))],
  caption: [
    Generalization to higher percentages of walls with guaranteed achievability.
  ],
)

Because walls are chosen from all possible positions IID, some configurations
may wall off either the key or the door. In order to remove this confounder, we
ran a set of experiments in which we train the agent in the same 10% wall
setting as before, but evaluate it on a set of configurations that guarantee the
reachability of both goals. Specifically, we generate procedural mazes in which
all cells of the grid are reachable and sample some percentage of the walls in
the maze. As figure demonstrates, this widens the performance gap between our
method and the AD baseline.

==== Model Accuracy
<model-accuracy>
In order to acquire a better understanding of the model's ability to in-context
learn, we plotted model accuracy in the generalization to 10% walls setting.
Note that while the percentages of walls in the training and evaluation setting
are the same, the exact wall placements are varied during training and the
evaluation wall placements are withheld, so that the model must infer them from
context. In figure , we measure the model's

prediction accuracy of termination signals \(labeled "done / not done"), of next
observations \(labeled "observation"), and of rewards \(labeled
"reward"). These predictions start near optimal, since the agent can rely on
priors, that most timesteps do not terminate, that most transitions result in
successful movement \(no wall), and that the reward is 0. However, we also
measure prediction accuracy for these rare events: the line labeled "done"
measures termination-prediction accuracy exclusively on terminal timesteps; the "positive
reward" line measures reward-prediction accuracy exclusively on timesteps with
positive reward; and the "wall" line measures accuracy on timesteps when the
agent's movement is obstructed by a random wall. As figure demonstrates, even
for these rare events, the model rapidly recovers accuracy near 100%.

==== Contribution of Model Error to Performance
<contribution-of-model-error-to-performance>
#figure(
  [#box(image("figures/adpp/model-noise.png"))],
  caption: [
    Impact of model error on performance, measured by introducing noise into each
    component of the model's predictions.
  ],
)

While figure indicates that our model generally achieves high accuracy in these
simple domains, we nevertheless wish to understand the impact of a suboptimal
model on RL performance. To test this, we introduced noise into different
component of the model's predictions. In figure , we note that performance is
fairly robust to noise in the termination predictions, but very sensitive to
noise in the reward predictions. Encouragingly, the model is demonstrates
reasonable performance with as much as 20% noise in the observation predictions.
Also, as indicates, the method is quite robust to noise in the action model. We
also note that AD's sensitivity to noise in the policy explains its lower
performance in many of the settings previously discussed.

#figure(
  [#box(image("figures/adpp/policy-noise-timestep.png"))],
  caption: [
    Impact of policy noise on performance, measured by interpolating policy logits
    with uniform noise.
  ],
)

==== Data Scaling Properties
<data-scaling-properties>
#figure([#figure(
    [#box(image("figures/adpp/less-source-data-time-timestep.png"))],
    caption: [
      Impact of scaling the training data along the IID dimension.
    ],
  )

  #figure(
    [#box(image("figures/adpp/less-source-data-iid-timestep.png"))],
    caption: [
      Impact of scaling the length of training of the source algorithm.
    ],
  )

], outlined: false)

Finally, we examined the impacts of scaling the quantity of data that our model
was trained on. In figure , we scale the quantity of the training data along the
IID dimension, with the $x$-axis measuring the number of source algorithm
histories in the training data scaled according to the equation $256 times 2^x$.
In figure , we scale the length for which each source algorithm is trained, with
the $x$-axis measuring the number of timesteps of training scaled according to
the same equation. This result was surprising, as we expected AD to be
#emph[more] sensitive to reduced training time, since that algorithm is more
dependent on demonstration of the optimal policy. Nevertheless we note that our
method outperforms AD in all data regimes.

== Conclusion
<conclusion>
This chapter presents an approach for combining ICPI with AD. The resulting
method scales to more complex settings than those explored in the previous
chapter. Moreover, the method significantly outperforms vanilla AD in a wide
variety of settings. For the final version of this thesis, we intend to test
this method on more complex domains, especially those involving simulated
robotics. We also intend to evaluate more baselines, especially those from the
traditional meta-learning literature like RL$""^2$ #cite(<duan_rl2_2016>).

// = Dummy <chap:pi>
// #bibliography("main.bib", style: "association-for-computing-machinery")