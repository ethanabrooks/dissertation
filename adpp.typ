#import "math.typ": *
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms
#import "@preview/tablex:0.0.7": tablex, rowspanx, colspanx

#show: show-algorithms

= #text(hyphenate: false, "Algorithm Distillation + Model-Based Planning")
== Introduction <sec:introduction-adpp>
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
methodology to choose actions on the downstream evaluation task. Concretely, on
each timestep, we use our learned model to perform multiple rollouts, each
starting from a different action in the action space. We use these rollouts to
estimate state-action values. Our behavior policy (as opposed to the policy used
in the rollouts) chooses the action corresponding to the highest estimate. As we
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
only on local information whereas a policy relies on non-local information. It
is worth noting that model-based planning gains this advantage at the cost of
compound error --- the tendency for any approximate model to accumulate error in
an auto-regressive chain, compounding the errors of new predictions with the
errors of prior predictions on which they are conditioned. Why should we expect
the benefit to outweigh the cost?

We argue that a function approximator can only learn a policy one of two ways:
either it learns the underlying logic of the policy, or it memorizes the policy
without distilling this logic --- a strategy which does not generalize. However,
this underlying logic entails some kind of implicit value estimation, which
entails implicit modeling of the environment. In principle, these implicit
operations are susceptible to compound error accumulation in much the same way
as explicit model-based planning methods. Therefore the switch to model-based
planning does not actually introduce compound error as a new cost. Meanwhile,
model-based planning does eliminate the possibility of policy memorization,
since actions and values are not inferred directly but computed. Our conclusion
is that the net benefit of model-based planning for generalization is positive.

We tested this hypothesis in two sets of domains, a gridworld with randomly
spawned walls, and a simulated robotics domain implemented in Mujoco
#cites(<todorov2012mujoco>). In the gridworld setting, we found that the
model-based approach consistently outperformed a direct policy adaptation
approach, even as we varied the difficulty of generalization and the quantity of
training data. We reach a similar conclusion in the robotics domain, though the
results are less extensive.

== Background
<sec:background>

For a review of Markov Decision Processes (MDPs) and model-based-planning, see
@sec:markov-decision-processes and @sec:model-based-planning, respectively.

==== Transformers
<transformers>
Transformers #cite(<vaswani2017attention>) are a class of neural networks which
map an input sequence to an output sequence of the same length and utilize a
mechanism known as "self-attention." This mechanism maps the $i^"th"$ element of
the input sequence to a key vector $k_i$, a value vector $v_i$, and query vector $q_i$.
The output, per index, of self-attention is a weighted sum of the value vectors:
$ upright("Attention") lr((q_i comma k comma v)) eq sum_j upright("softmax")
lr((q_i k_j slash sqrt(D))) v_j $
where $D$ is the dimensionality of the vectors. The softmax is applied across
all inputs so that
$sum_j upright("softmax") lr((q_i k_j slash sqrt(D))) eq 1$. A typical
transformer applies this self-attention operation multiple times, interspersing
linear projections and layer-norm operations between each application. A #emph[causal] transformer
applies a mask to the attention operation which prevents the model from
attending from the $i^"th"$ element to the $j^"th"$ element if $i < j$, that is,
if the $j^"th"$ element appears later in the sequence. Intuitively, this
prevents the model from conditioning inferences on future information.

==== Algorithm Distillation <sec:algorithm-distillation>
Algorithm Distillation #cite(<laskin2022context>) is a method for distilling the
logic of a source RL algorithm into a causal transformer. The method assumes
access to a dataset of "learning histories" generated by the source algorithm,
which comprise observation, action, reward sequences spanning the entire course
of learning, starting with initial random behavior and ending with fully
optimized behavior. The transformer is trained to predict actions given the
entire history of behavior or some large subset of it. Optimal prediction of
these actions requires the model to capture not only the current policy but also
the improvements to that policy applied by the source algorithm. Auto-regressive
rollouts of the distilled model, in which actions predicted by the model are
actually used to interact with the environment and then cycled back into the
model input, demonstrate improvement of the policy, often to the point of near
optimality, without any adjustment to the model's parameters. This indicates
that the model does, in fact, capture some policy improvement logic from the
source algorithm.

== Method <sec:method>
In this section, we describe the details of our proposed algorithm, which we
call #ADPP-full-name (#ADPP). We assume a dataset of $N$ trajectories generated
by interactions with an environment:

$ Dataset := ( (Obs^n_0, Act^n_0, Rew^n_0, Ter^n_0, dots, Obs^n_T, Act^n_T,
Rew^n_T, Ter^n_T) sim Policy_n )_(n=1)^N $

with $Obs^n_t$ referring to the $t^"th"$ observation in the $n^"th"$
trajectory, $Act^n_t$ to the $t^"th"$ action, $Rew^n_t$ to the
$t^"th"$ reward, and $Ter^n_t$ to the $t^"th"$ termination boolean. Each
trajectory corresponds with a single task $Task$ and a single policy
$Policy$, but the dataset contains as many as $N$ tasks and policies. We also
introduce the following nomenclature for trajectory histories:

$ History^n_t := (Act^n_(t-Recency), Rew^n_(t-Recency), Ter^n_(t-Recency),
Obs^n_(t-Recency+1), dots, Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t) $ <eq:history>

where $Recency$ is a fixed hyperparameter.

== Model Training <sec:model-training>
Given such a dataset, we may train a world-model with a negative log likelihood
\(NLL) loss:

$ Loss_theta := -sum_(n=1)^N sum_(t=1)^(T-1) log Prob_theta (Act^n_t |
History^n_t) + log Prob_theta (Rew^n_t, Ter^n_t, Obs^n_(t+1) | History^n_t,
Act^n_t) $ <eq:loss-adpp>

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
<sec:downstream-evaluation>
#algorithm-figure(
  {
    import "algorithmic.typ": *
    algorithm(..Function(
      $QValue(History_t, Act)$,
      State($u gets 1$, comment: "time step for rollout"),
      State($Act^u gets Act$),
      ..While("termination condition not met", State(
        $Rew^u, Ter^u, Obs^(u+1) sim Prob_theta (dot.c|History_t, Act^u)$,
        comment: "Model reward, termination, next observation",
        label: <line:model-world>,
      ), State(
        $a^(u+1) sim Prob_theta (dot.c|History_t, Rew_t, Ter_t, Obs_(t+1))$,
        comment: "Model policy",
        label: <line:model-policy>,
      ), State($u gets u + 1$, comment: "Append predictions to history.")),
      Return($sum_(u=0)^t gamma^(u-1) Rew_u$, label: <line:return>),
    ))
  },
  caption: [ Estimating Q-values with monte-carlo rollouts. ],
  placement: top,
) <alg:downstream-evaluation>

Our approach to choosing actions during the downstream evaluation is as follows.
For each action $Act$ in a set of candidate actions (either our complete action
space or some subset thereof), we compute a state-action value estimate
$QValue(History_t, Act)$, where $History_t$ is the history defined in
@eq:history. We do this by modeling a rollout conditioned on the history $History_t$,
and from $Act$. Modelling the rollout is a cycle of sampling values from the
model and feeding them back into the model auto-regressively. First we sample
$(Rew^u, Ter^u, Obs^(u+1))$ (@line:model-world in @alg:downstream-evaluation).
Based on this prediction, we sample
$Act^(u+1)$ (@line:model-policy). Adding this to the input allows us to sample
$(Rew^(u+1), Ter^(u+1), Obs^(u+2))$, repeating this cycle until some termination
condition is reached. For example, we might terminate the process once
$Ter^u$ is true, or once the rollout reaches some maximum length.

The final step is to choose an action. Having modelled the rollout, we compute
the value estimate as the discounted sum of rewards in the rollout
(@line:return). Repeating this process of modelling rollouts and computing value
estimates for each action in a set of candidate actions, we simply choose the
action with the highest value estimate:
$arg max_(Act in Actions) QValue(History_t, Act)$.

== Policy improvement
<sec:policy-improvement>
If the downstream task is sufficiently dissimilar to the tasks in the training
dataset, or if the training dataset does not contain optimal policies, then it
is unlikely that the procedure described above will initially yield an optimal
policy. Some method for policy improvement will be necessary.

==== Policy Iteration
Our method satisfies this requirement by implementing a form of policy
iteration. To see this, first observe that our model is always trained to map a
behavior history drawn from a single policy to actions drawn from the same
policy. A fully trained model will therefore learn to match the distribution of
actions in its output to the distribution of actions in its input. Since our
rollouts are conditioned on histories drawn from our behavior policy, the
rollout policy will approximately match this policy. Our value estimates will
therefore be conditioned on the current behavior policy. However, by choosing
the action corresponding to $arg max_(Act in Actions) QValue^Policy (History_t, Act)$,
our behavior policy always improves on $Policy$, the policy on which $QValue^Policy (History_t, Act)$ is
conditioned, a consequence of the policy improvement theorem. Thus, each time an
action is chosen using this $arg max$ method, our behavior policy improves on
itself.

Walking through this process step by step, suppose $Policy^n$ is some policy
that we use to behave. We collect a trajectory containing actions drawn from
this policy. When we perform rollouts, we condition on this trajectory and the
rollout policy simulates $Policy^n$. Assuming that this simulation is accurate
as well as the world model, our value estimate will be an unbiased monte carlo
estimate of $QValue^(Policy^n) (History_t, Act)$ for any action $Act$. Then we
act with policy $Policy^(n+1) := arg max_(Act in Actions) QValue^(Policy_n) (History_t, Act)$.
But
$Policy^(n+1)$ is at least as good as $Policy^n$. Using the same reasoning, $Policy^(n+2)$ will
be at least as good as
$Policy^(n+1)$, and so on. Note that in our implementation, we perform the $arg max$ at
each step, rather than first collecting a full trajectory with a single policy.

==== Algorithm Distillation
Our setting is almost identical to Algorithm Distillation (AD)
#cite(<laskin2022context>), if we include the assumption that trajectories
include a full learning history. Rather than competing with AD, our method is
actually complementary to it. If the input to our transformer is a sufficiently
long history of behavior, then the rollout policy will not only match the input
policy but actually improve upon it, as demonstrated in that paper. Then the
procedure described in @alg:downstream-evaluation for estimating $Q$-values will
actually condition these estimates on a policy $Policy'_n$ that is at least as
good as the input policy
$Policy_n$. Then $V^(Policy_(n)) <= V^(Policy'_n) <= V^(Policy_(n+1))$.
Therefore each step of improvement actually superimposes the two improvement
operators, one from the $arg max$ operator, the other from AD.

=== Extension to Continuous Actions <sec:continuous-actions>
When evaluating #ADPP on continuous action domains, the formulation we have
described must be modified, since it is not possible to iterate over the action
space. Instead, we sample a fixed number of actions from $Prob_theta (dot.c|History_t, Rew_t, Ter_t, Obs_(t+1))$ and
for each sampled action, perform a rollout per @alg:downstream-evaluation. This
step does not preserve the policy improvement guarantees
(@sec:policy-improvement), but works well enough in practice.

=== Beam search <sec:beam-search>
The algorithm that we have described can be further augmented with beam Search,
similar to #cite(<janner2021sequence>, form: "prose"). While this proved useful
in only one of the domains that we evaluated, we describe it here for the sake
of completeness. Using beam search requires learning a value function which is
used for pruning the tree. We augment @eq:loss-adpp with the following term:

$ Loss'_theta := Loss_theta + sum_(n=1)^N sum_(t=1)^(T-1) log Prob_theta (sum_(u=t)^T gamma^(T-u) Rew_u | History^n_t) $

When conditioning $History^n_t$, the model $Prob_theta$ outputs two predictions,
one corresponding to the next action, the other corresponding to the value of
the current state. For the latter, we use the discounted cumulative sum of
empirical reward as a target.

The procedure for estimating value with beam search is as follows. On each step
of the rollout procedure described in @alg:downstream-evaluation, we sample $N$ actions
(instead of just one per rollout). Instead of a series of parallel chains, as in
the original algorithm, this procedure would result in an expanding tree with
arity $N$. For computational tractability, we therefore rank the paths by value
(as estimated by $Prob_theta$) and discard the bottom $(N - 1) / N$ so that the
number of active pathways per rollout step does not increase. Ranking is
performed across all rollouts, so all the descendants of a given node may
eventually be pruned.

== Related Work
A paper that strongly influenced this work is Trajectory Transformer #cite(<janner2021sequence>),
from which we borrow much of our methodology. We distinguish ourselves from that
work by focusing on in-context learning in partially observed multi-task
domains, and through the incorporation of Algorithm Distillation #cite(<laskin2022context>).

Several recent works have explored the use of in-context learning to adapt
transformer-based agents to new tasks. #cite(<raparthy2023generalization>) study
the properties that are conducive to generalization in these kinds of agents,
especially highlighting "burstiness" #cite(<chan2022data>) and "multi-trajectory"
inputs -- inputs containing multiple episodes from the same task, as used in #cite(<laskin2022context>, form: "prose") and
in this work. #cite(<lee2023supervised>, form: "prose") propose an approach
similar to AD, but instead of predicting the _next_ action, they directly
predict the _optimal_ action. They demonstrate that this gives rise to similar
forms of in-context learning and outperforms AD on several tasks.
#cite(<pinon2022model>, form: "prose") train a dynamics model, similar to this
work, and execute tree search, though unlike this work, they use a fixed policy.

Transformers have also been studied extensively in the capacity of world models. #cite(<micheli2022transformers>, form: "prose") train
a transformer world-model using methods similar to our own and demonstrate that
training an agent to optimize return within this model is capable of
significantly improving sample-complexity on the Atari 100k benchmark. #cite(<robine2023transformer>, form: "prose") augment
this architecture with a variational autoencoder for generating compact
representations of the observations. "TransDreamer" #cite(<chen2022transdreamer>) directly
emulates Dreamer V2 #cite(<hafner2020mastering>) adjusting that algorithm to use
transformers in place of GRUs to capture recurrence.

== Experiments
<experiments>
==== Domains
<domains>
In this work, we chose to focus on partially observable domains. This ensures
that both the initial policy and the initial model in our downstream domain will
be suboptimal, since the true dynamics or reward function cannot be inferred
until the agent has gathered experience. Recovering the optimal policy will
coincide with model improvement. Model improvement will occur as the agent
collects experience and the transformer context is populated with transitions
drawn from the current dynamics and reward functions. Policy improvement relies
on the mechanisms detailed in the previous sections. One hypothesis that our
experiments test is whether these learning processes can successfully happen
concurrently.

Our first set of experiments occur within a discrete, partially observable,
$5 times 5$ grid world. The agent has four actions, up, left, down, and right.
For each task, we spawn a "key" and a "door" in a random location. The agent
receives a reward of 1 for visiting the key location and then a reward of 1 for
visiting the door. The agent observes only its own position and must infer the
positions of the key and the door from the history of interactions which the
transformer prompt contains. The episode terminates when the agent visits the
door, or after 25 time-steps.

==== Baselines
<baselines>
We compare our method with two baselines: vanilla AD and
"Ground-Truth" #ADPP. The latter is identical to our method, but uses a
ground-truth model of the environment, which is free of error. The comparison
with AD highlights the contribution of the model-based planning component of our
method. The comparison with Ground-Truth #ADPP establishes an approximate upper
bound for our method and distinguishes the contribution of model error to the RL
metrics that we record.

=== Results
<results>

==== Evaluation on Withheld Goals
<evaluation-on-withheld-goals>

<evaluation-on-withheld-wall-configurations>
#figure(image("figures/adpp/no-walls.png", width: 110%), caption: [
  Evaluation on withheld location pairs.
], placement: top) <fig:no-walls>

In our first experiment, we evaluate the agent on a set of withheld key-door
pairs, which we sample uniformly at random (10% of all possible pairs) and
remove from the training set. As @fig:no-walls indicates, our algorithm
outperforms the AD baseline both in time to converge and final performance. We
attribute this to the fact that our method's downstream policy directly
optimizes expected return, choosing actions that correspond to the highest value
estimate. In contrast, AD's policy only maximizes return by proxy — maximizing
the probability of the actions of a source algorithm which in turn maximizes
expected return. This indirection contributes noise to the downstream policy
through modeling error. Moreover, we note that our method completely recovers
the performance of the ground-truth baseline, though its speed of convergence
lags slightly, due to the initial exploration phase in which the model learns
the reward function through trial and error.

#figure(
  image("figures/adpp/unseen-goals.png", width: 300pt),
  caption: [ Evaluation on fully withheld locations. ],
  placement: top,
) <fig:unseen-goals>

Next, we increase the generalization challenge by holding out key and door
locations entirely. During training of the source algorithm, we never place keys
or doors in the upper-left four cells of the grid. During evaluation, we place
both keys and doors exclusively within this region. As @fig:unseen-goals
demonstrates, AD generalizes poorly in this setting, on average discovering only
one of the two goals. In contrast, our method maintains relatively high
performance. We attribute this to the fact that our method learns low-level
planning primitives (the reward function), which generalize better than
high-level abstractions like a policy. As we argued in @sec:introduction-adpp,
higher-level abstractions are prone to memorization since they do not perfectly
distill the logic which produced them.

==== Evaluation on Withheld Wall Configurations

In addition to evaluating generalization to novel reward functions, we also
evaluated our method's ability to generalize to novel dynamics. We did this by
adding walls to the grid world, which obstruct the agent's movement. During
training we placed the walls at all possible locations, sampled IID, with 10%
probability. During evaluation, we tested the agent on equal or higher
percentages of wall placement. As indicated by
@fig:generalization-to-more-walls, our method maintains performance and nearly
matches the ground-truth version, while AD's performance rapidly degrades. Again
we attribute this to the tendency of lower-level primitives to generalize better
than higher-level abstractions.

#figure(
  [#box(image("figures/adpp/generalization-to-more-walls-timestep.png"))],
  caption: [
    Generalization to higher percentages of walls.
  ],
  placement: top,
) <fig:generalization-to-more-walls>

Because walls are chosen from all possible positions IID, some configurations
may wall off either the key or the door. In order to remove this confounder, we
ran a set of experiments in which we train the agent in the same 10% wall
setting as before, but evaluate it on a set of configurations that guarantee the
reachability of both goals. Specifically, we generate procedural mazes in which
all cells of the grid are reachable and sample some percentage of the walls in
the maze. As @fig:generalization-to-more-walls-with-guaranteed-achievability
demonstrates, this widens the performance gap between our method and the AD
baseline.

#figure(
  [#box(image("figures/adpp/more-walls-achievable-timestep.png"))],
  caption: [
    Generalization to higher percentages of walls with guaranteed achievability.
  ],
  placement: bottom,
) <fig:generalization-to-more-walls-with-guaranteed-achievability>

==== Model Accuracy
<model-accuracy>

#figure(
  image("figures/adpp/model-accuracy.png", width: 300pt),
  caption: [Accuracy of model predictions over the course of an evaluation rollout.],
  placement: top,
) <fig:model-accuracy>

In order to acquire a better understanding of the model's ability to in-context
learn, we plotted model accuracy in the generalization to 10% walls setting.
Note that while the percentages of walls in the training and evaluation setting
are the same, the exact wall placements are varied during training and the
evaluation wall placements are withheld, so that the model must infer them from
context. In @fig:model-accuracy, we measure the accuracy of the model's
prediction of termination signals (labeled "done / not done"), of next
observations (labeled "observation"), and of rewards (labeled
"reward"). These predictions start near optimal, since the agent can rely on
priors: that most timesteps do not terminate, that most transitions result in
successful movement (no wall), and that the reward is 0. However, we also
measure prediction accuracy for these rare events: the line labeled "done"
measures termination-prediction accuracy for terminal timesteps only; the "positive
reward" line measures reward-prediction accuracy on timesteps with positive
reward; and the "wall" line measures accuracy on timesteps when the agent's
movement is obstructed by a random wall. As @fig:model-accuracy demonstrates,
even for these rare events, the model rapidly recovers accuracy near 100%.

==== Contribution of Model Error to Performance
<contribution-of-model-error-to-performance>
#figure(
  [#box(image("figures/adpp/model-noise.png", width: 110%))],
  caption: [
    Impact of model error on performance, measured by introducing noise into each
    component of the model's predictions.
  ],
  placement: top,
) <fig:model-noise>

While @fig:model-noise indicates that our model generally achieves high accuracy
in these simple domains, we nevertheless wish to understand the impact of a
suboptimal model on RL performance. To test this, we introduced noise into
different components of the model's predictions. In @fig:model-noise, we note
that performance is fairly robust to noise in the termination predictions, but
very sensitive to noise in the reward predictions. Encouragingly, the model
demonstrates reasonable performance with as much as 20% noise in the observation
predictions. Also, as indicated, the method is quite robust to noise in the
action model. We also note that AD's sensitivity to noise in the policy explains
its lower performance in many of the settings previously discussed.

==== Data Scaling Properties
<data-scaling-properties>
#figure(
  [#box(image("figures/adpp/less-source-data-iid-timestep.png"))],
  caption: [
    Impact of scaling the length of training of the source algorithm.
  ],
  placement: bottom,
)<fig:less-source-data-iid>

We also examined the impacts of scaling the quantity of data that our model was
trained on. In @fig:less-source-data-iid, we scale the quantity of the training
data along the IID dimension, with the $x$-axis measuring the number of source
algorithm histories in the training data scaled according to the equation $256 times 2^x$.
In @fig:less-source-data-time, we scale the length for which each source
algorithm is trained, with the $x$-axis measuring the number of timesteps of
training scaled according to the same equation. This result was surprising, as
we expected AD to be
#emph[more] sensitive to reduced training time, since that algorithm is more
dependent on demonstration of the optimal policy. Nevertheless, we note that our
method outperforms AD in all data regimes.
#figure(
  [#box(image("figures/adpp/less-source-data-time-timestep.png"))],
  caption: [
    Impact of scaling the training data along the IID dimension.
  ],
  placement: top,
)<fig:less-source-data-time>

=== Continuous-State and Continuous-Action Domains
Finally, we evaluate the ability of #ADPP to learn in domains with continuous
states and actions. In order to adapt #ADPP to infinitely large action spaces,
we use the sampling technique described in @sec:continuous-actions. In the
experiments that follow, the number of actions sampled at the beginning of the
rollouts is 128.

==== Sparse Point Environment
In the "Sparse Point" environment, a point spawns randomly on a half circle. The
agent does not observe the point and has to discover it through exploration. The
agent receives reward for occupying coordinates within a fixed radius of the
goal. The agent's observation space consists of 2d position coordinates and its
action space consists of 2d position deltas. The Sparse Point environment tests
the ability of the agent to explore efficiently, since an agent that
concentrates its exploration on the half-circle will significantly outperform
one that explores all positions with equal probability.

#figure(
  grid(
    columns: (auto, auto, auto),
    [#figure([#box(image("figures/adpp/point-env.png", height: 100pt))], caption: [
        Evaluation on the "Sparse-Point" environment.
      ]) <fig:point-env>],
    [#figure(
        [#box(image("figures/adpp/cheetah-dir.png", height: 100pt))],
        caption: [
          Evaluation on the "Half-Cheetah Direction" domain.
        ],
      ) <fig:cheetah-dir>],
    [#figure(
        [#box(image("figures/adpp/cheetah-vel.png", height: 100pt))],
        caption: [
          Evaluation on the "Half-Cheetah Velocity" domain.
        ],
      ) <fig:cheetah-vel>],
  ),
  outlined: false,
  placement: top,
  kind: "none",
  supplement: none,
)

As @fig:point-env demonstrates, vanilla AD fares quite poorly in this setting.
Of the 20 seeds in the diagram, only two discover the goal and only one returns
to it consistently. The AD agent either explores randomly in vicinity of the
origin --- emulating policies observed early in the source data --- or commits
arbitrarily to a point on the arc and remains in its vicinity --- emulating
later policies, but ignoring the lack of experienced reward. The Sparse Point
environment highlights a weakness in vanilla AD. During training, the source
algorithm --- which uses one agent per task --- can memorize the location of the
goal. It therefore never exhibits Bayes-optimal exploration patterns for AD to
imitate.

Why should we expect #ADPP to perform better? For the same reasons that cause
vanilla AD to fail, we should not expect the simulated rollouts of #ADPP to
perform Bayes-optimal exploration. However, before the model experiences reward,
its reward predictions will reflect the prior, which has support on the
semi-circle and nowhere else. Therefore any rollouts that do not lead to the
semi-circle should result in a simulation of zero cumulative reward.

In order for #ADPP to recover Bayes optimal exploration in the Sparse Point
environment, two random events must co-occur: some of the rollouts must lead to
the semi-circle, and the model must anticipate reward there, without having
necessarily experienced it. In practice, this does not happen consistently. The
rollout policy often degenerates into random dithering and the reward model
often predicts no reward at all. We found that the key to eliciting consistent
performance from #ADPP was to perform beam-search as described in
@sec:beam-search. This effectively increases the opportunities for rollouts to
lead to the arc and for the reward model to anticipate reward there. For
example, paths that do not lead to the arc can be pruned entirely, while those
that do can benefit from node-expansion. As @fig:point-env demonstrates, both
beam-search methods significantly outperform vanilla AD and #ADPP.

==== Half-Cheetah Environments

In our final set of environments, we explore two variants on the well-known
Mujoco "Half-Cheetah" environment. This environment uses a 2D two-legged
embodiment. In order to instantiate a multi-task problem, we vary the reward
function: for the "Half-Cheetah Direction" environment, we instantiate two
tasks, one which rewards the agent for forward movement of the Cheetah and one
that rewards it for backward movement. For the "Half-Cheetah Velocity"
environment, we choose a target velocity per task. The agent receives a reward
penalty for the difference between its current velocity and the target. Per the
original Half-Cheetah environment, the agent also receives a control reward that
penalizes large actions. As @fig:cheetah-dir and @fig:cheetah-vel demonstrate, #ADPP outperforms
vanilla AD on both domains.

== Conclusion
<conclusion>
This chapter presents an approach for combining ICPI with AD. The resulting
method scales to more complex settings than those explored in the previous
chapter. Moreover, the method significantly outperforms vanilla AD in a variety
of settings.

// = Dummy <chap:pi>
// == Dummy <sec:model-based-planning>
// == Dummy <sec:markov-decision-processes>
// #bibliography("main.bib", style: "american-society-of-civil-engineers")