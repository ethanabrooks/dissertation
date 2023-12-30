#import "math.typ": *
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms
#import "@preview/big-todo:0.2.0": todo
#import "@preview/tablex:0.0.7": tablex, rowspanx, colspanx

#show: show-algorithms

#set heading(numbering: "1.1")

= In-Context Policy Iteration <chap:pi>
In the context of Large Language Models (LLMs), in-context learning describes
the ability of these models to generalize to novel downstream tasks when
prompted with a small number of exemplars
#cites(<lu_pretrained_2021>, <brown_language_2020>). The introduction of the
Transformer architecture #cite(<vaswani_attention_2017>) has significantly
increased interest in this phenomenon, since this architecture demonstrates much
stronger generalization capacity than its predecessors
#cite(<chan_data_2022>). Another interesting property of in-context learning in
the case of large pre-trained models (or "foundation models") is that the models
are not directly trained to optimize a meta-learning objective, but demonstrate
an emergent capacity to generalize (or at least specialize) to diverse
downstream task-distributions
#cites(
  <lu_pretrained_2021>,
  <brown_language_2020>,
  <chan_data_2022>,
  <wei_emergent_2022>,
).
A litany of existing work
has explored methods for applying this remarkable capability to downstream tasks
(see @sec:related), including Reinforcement Learning (RL). Most work in this
area either (1) assumes access to expert demonstrations --- collected either from
human experts
#cites(<huang_language_2022>, <baker_video_2022>), or domain-specific
pre-trained RL agents
#cites(
  <chen_decision_2021>,
  <lee_multi-game_2022>,
  <janner_offline_2021>,
  <reed_generalist_2022>,
  <xu_prompting_2022>,
) --- or (2) relies on gradient-based methods --- e.g. fine-tuning of the
foundation models parameters as a whole
#cites(<lee_multi-game_2022>, <reed_generalist_2022>, <baker_video_2022>) or
newly training an adapter layer or prefix vectors while keeping the original
foundation models frozen
#cites(<li_prefix-tuning_2021>, <singh_know_2022>)
#cites(<karimi_mahabadi_prompt-free_2022>).

Our work presents an algorithm, In-Context Policy Iteration (ICPI) which relaxes
these assumptions. ICPI is a form of policy iteration in which the prompt
content is the locus of learning. Because our method operates on the prompt
itself rather than the model parameters, we are able to avoid gradient methods.
Furthermore, the use of policy iteration frees us from expert demonstrations
because suboptimal prompts can be improved over the course of training.

We illustrate the algorithm empirically on six small illustrative RL
tasks --- _chain, distractor-chain, maze, mini-catch, mini-invaders_, and
_point-mass_ --- in which the algorithm very quickly finds good policies. We also
compare five pretrained Large Language Models (LLMs), including two different
size models trained on natural language --- OPT-30B and GPT-J --- and three
different sizes of a model trained on program code --- two sizes of Codex as well
as InCoder. On our six domains, we find that only the largest model (the
`code-davinci-001` variant of Codex) consistently demonstrates learning.

#figure(
  image("figures/policy-iteration/action-selection.svg"),
  caption: [
    For each possible action $Actions(1), dots.h, Actions(n)$, the LLM generates a
    rollout by alternately predicting transitions and selecting actions. Q-value
    estimates are discounted sums of rewards. The action is chosen greedily with
    respect to Q-values. Both state/reward prediction and next action selection use
    trajectories from $Buffer$
    to create prompts for the LLM. Changes to the content of $Buffer$
    change the prompts that the LLM receives, allowing the model to improve its
    behavior over time.
  ],
  placement: top,
) <fig:q>

== Related Work <sec:related>

A common application of foundation models to RL involves tasks that have
language input, for example natural language instructions/goals
#cites(<garg_lisa_2022>, <hill_human_2020>) or text-based games
#cites(
  <peng_inherently_2021>,
  <singh_pre-trained_2021>,
  <majumdar_improving_2020>,
  <ammanabrolu_learning_2021>,
). Another approach encodes RL trajectories into token sequences, and processes
them with a foundation model, and passes the model outputs to deep RL
architectures
#cites(<li_pre-trained_2022>, <tarasov_prompts_2022>, <tam_semantic_2022>).
Finally, a recent set of approaches (which we will focus on in this Related Work
section) treat RL as a sequence modeling problem and use the foundation models
itself to predict states or actions. In this related work section, we will focus
a third set of recent approaches that treat reinforcement learning (RL) as a
sequence modeling problem and utilize foundation models for state prediction,
action selection, and task completion. We will organize our survey of these
approaches based on how they elicit these RL-relevant outputs from the
foundation models. In this respect the approaches fall under three broad
categories: learning from demonstrations, specialization (via training or
finetuning), and context manipulation (in-context learning).

=== Learning from demonstrations
Many recent sequence-based approaches to reinforcement learning use
demonstrations that come either from human experts or pretrained RL agents. For
example, #cite(<huang_language_2022>, form: "prose") use a frozen LLM as a planner for everyday
household tasks by constructing a prefix from human-generated task instructions,
and then using the LLM to generate instructions for new tasks. This work is
extended by #cite(<huang_inner_2022>, form: "prose"). Similarly,
#cite(<ahn_as_2022>, form: "prose") use a value function that is trained on human
demonstrations to rank candidate actions produced by an LLM.
#cite(<baker_video_2022>, form: "prose") use human demonstrations to train the foundation model
itself: they use video recordings of human Minecraft players to train a
foundation model to play Minecraft. Works that rely on pretrained RL agents
include #cite(<janner_offline_2021>, form: "prose") who train a "Trajectory Transformer" to
predict trajectory sequences in continuous control tasks by using trajectories
generated by pretrained agents, and #cite(<chen_decision_2021>, form: "prose"), who use a
dataset of offline trajectories to train a "Decision Transformer" that predicts
actions from state-action-reward sequences in RL environments like Atari. Two
approaches build on this method to improve generalization:
#cite(<lee_multi-game_2022>, form: "prose") use trajectories generated by a DQN agent to train
a single Decision Transformer that can play many Atari games, and
#cite(<xu_prompting_2022>, form: "prose") use a combination of human and artificial
trajectories to train a Decision Transformer that achieves few-shot
generalization on continuous control tasks. #cite(<reed_generalist_2022>, form: "prose") take
task-generality a step farther and use datasets generated by pretrained agents
to train a multi-modal agent that performs a wide array of RL (e.g. Atari,
continuous control) and non-RL (e.g. image captioning, chat) tasks.

Some of the above works include non-expert demonstrations as well.
#cite(<chen_decision_2021>, form: "prose") include experiments with trajectories generated by
random (as opposed to expert) policies. #cite(<lee_multi-game_2022>, form: "prose") and
#cite(<xu_prompting_2022>, form: "prose") also use datasets that include trajectories generated
by partially trained agents in addition to fully trained agents. Like these
works, our proposed method (ICPI) does not rely on expert demonstrations---but
we note two key differences between our approach and existing approaches.
Firstly, ICPI only consumes self-generated trajectories, so it does not require
any demonstrations (like #cite(<chen_decision_2021>, form: "prose") with random trajectories,
but unlike #cite(<lee_multi-game_2022>, form: "prose"), #cite(<xu_prompting_2022>, form: "prose"), and the
other approaches reviewed above). Secondly, ICPI relies primarily on in-context
learning rather than in-weights learning to achieve generalization (like
#cite(<xu_prompting_2022>, form: "prose"), but unlike #cite(<chen_decision_2021>, form: "prose") \&
#cite(<lee_multi-game_2022>, form: "prose")). For discussion about in-weights vs. in-context
learning see #cite(<chan_data_2022>, form: "prose").

=== Gradient-based training \& finetuning on RL tasks
Many approaches that use foundation models for RL involve specifically training
or fine-tuning on RL tasks. For example,
#cite(<janner_offline_2021>, form: "prose"), #cite(<chen_decision_2021>, form: "prose"),
#cite(<lee_multi-game_2022>, form: "prose"),
#cite(<xu_prompting_2022>, form: "prose"),
#cite(<baker_video_2022>, form: "prose"),
and
#cite(<reed_generalist_2022>, form: "prose")
all use models that are trained from scratch on tasks of interest, and
#cites(<singh_know_2022>, form: "prose"), #cite(<ahn_as_2022>, form: "prose"), and #cite(<huang_inner_2022>, form: "prose") combine frozen
foundation models with trainable components or adapters. In contrast,
#cite(<huang_language_2022>, form: "prose") use frozen foundation models for planning, without
training or fine-tuning on RL tasks. Like #cite(<huang_language_2022>, form: "prose"), ICPI
does not update the parameters of the foundation model, but relies on the frozen
model's in-context learning abilities. However, ICPI gradually builds and
improves the prompts within the space defined by the given fixed text-format for
observations, actions, and rewards (in contrast to #cite(<huang_language_2022>, form: "prose"),
which uses the frozen model to select good prompts from a given fixed library of
goal/plan descriptions).

=== In-Context learning
Several recent papers have specifically studied in-context learning.
#cite(<laskin2022context>, form: "prose") demonstrates an approach to performing in-context
reinforcement learning by training a model on complete RL learning histories,
demonstrating that the model actually distills the improvement operator of the
source algorithm. #cite(<min_rethinking_2022>, form: "prose") demonstrates that LLMs can
learn in-context, even when the labels in the prompt are randomized,
problemetizing the conventional understanding of in-context learning and
showing that label distribution is more important than label correctness.
#cite(<chan_data_2022>, form: "prose") and #cite(<garg_what_2022>, form: "prose") provide analyses of the
properties that drive in-context learning, the first in the context of image
classification, the second in the context of regression onto a continuous
function. These papers identify various properties, including "burstiness,"
model-size, and model-architecture, that in-context learning depends on.
#cite(<chen_relation_2022>, form: "prose") studies the sensitivity of in-context learning to
small perturbations of the context. They propose a novel method that uses
sensitivity as a proxy for model certainty. Some recent work has explored
iterative forms of in-context learning, similar to our own. For example,
#cite(<shinn2023reflexion>, form: "prose") and #cite(<madaan2023self>, form: "prose") use iterative
self-refinement to improve the outputs of a large language model in a natural
language context. These approaches rely on the ability of the model to examine
and critique its own outputs, rather than using policy iteration as our method
does.

== Background
<sec:background>
==== Markov Decision Processes
<sec:markov-decision-processes>
A Markov Decision Process (MDP) is a problem formulation in which an agent
interacts with an environment through actions, receiving rewards for each
interaction. Each action transitions the environment from some state to some
other state and the agent is provided observations which depend on this state
(and from which the state can usually be inferred). MDPs can be "fully" or "partially"
observed. A fully observed MDP is defined by the property that the distribution
for each transition and reward is fully conditioned on the current observation.
In a partially observed MDP, the distribution depends on history — some
observations preceding the current one. The goal of reinforcement learning is to
discover a "policy" — a mapping from observations to actions — which maximizes
cumulative reward over time.

==== Model-Based Planning
<sec:model-based-planning>
A model is a function which approximates the transition and reward distributions
of the environment. Typically a model is not given but must be learned from
experience gathered through interaction with the environment. Model-Based
Planning describes a class of approaches to reinforcement learning which use a
model to choose actions. These approaches vary widely, but most take advantage
of the fact that an accurate model enables the agent to anticipate the
consequences of an action before taking it in the environment.

==== Policy Iteration
<sec:policy-iteration>

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

#let formatAction = (s) => text(fill: blue)[#s]
#let formatReward = (s) => text(fill: orange)[#s]
#let formatState = (s) => text(fill: green)[#s]
#let formatTermination = (s) => text(fill: eastern)[#s]
#let formatValue = (s) => text(fill: purple)[#s]

#algorithm-figure(
  {
    import "algorithmic.typ": *
    let Obs = formatState($Obs$)
    let Rew = formatReward($Rew$)
    let Act = formatAction($Act$)
    let QValue = formatValue($QValue$)
    let reward = formatReward("reward")
    let state = formatState("state")
    let termination = formatTermination("termination")
    algorithm(
      State([initialize $Buffer$], comment: [Replay Buffer]),
      ..
      While(
        [training],
        State([$Obs_0 gets$ Reset environment], comment: [Get initial #state]),
        ..
        For(
          [each timestep $t$ in episode],
          State(
            [ $formatAction(Act) gets arg max_Act QValue^Policy (Obs_t, Act) $ ],
            comment: [ Choose #formatAction("action") with highest #formatValue("value") ],
            label: <line:arg-max>,
          ),
          State(
            [ $Rew_t, Ter_t, Obs_(t+1) gets$ step environment with $Act$ ],
            comment: [ Receive #reward and next #state ],
          ),
          State(
            $Buffer gets Buffer union (Obs_t, Act_t, Rew_t, Ter_t)$,
            comment: [ Add interaction to replay buffer ],
          ),
        ),
      ),
    )
  },
  caption: [
    Training Loop
  ],
) <algo:train>
#algorithm-figure(
  {
    import "algorithmic.typ": *
    import "math.typ"
    let Act = formatAction($Act$)
    let Ter = formatTermination($Ter$)
    let Obs = formatState($Obs$)
    let Rew = formatReward($Rew$)
    let QValue = formatValue($QValue$)
    let reward = formatReward("reward")
    let state = formatState("state")
    let termination = formatTermination("termination")
    let value = formatValue("value")

    algorithm(
      ..Function(
        $Q(Obs_t, Act, Buffer)$,
        State($u gets t$),
        State($Obs^1 gets Obs_t$),
        State($Act^1 gets Act_t$),
        ..Repeat(
          State([$Buffer_Ter sim$ time-steps with action $Act^u$]),
          State(
            [$Ter^u sim LLM(Buffer_Ter, Obs^u, Act^u)$],
            comment: [ model #termination ],
          ),
          State(
            [$Buffer_Rew sim$ time-steps with action $Act^u$ and #termination $Ter^u$],
          ),
          State(
            [$Rew^u sim LLM(Buffer_Ter, Obs^u, Act^u)$],
            comment: [ model #reward ],
          ),
          State(
            [$Buffer_Obs sim$ time-steps with action $Act^u$ and #termination $Ter^u$],
          ),
          State(
            [$Obs^(u+1) sim LLM(Buffer_Obs, Obs^u, Act^u)$],
            comment: [ model #termination ],
          ),
          State([$Buffer_Act sim Recency$ recent trajectories]),
          State(
            [$Act^(u+1) sim LLM(Buffer_Act, Obs^(u+1))$],
            comment: [ model policy ],
          ),
          State($u gets u+1$),
          [$Ter^u$ is terminal],
          comment: [model predicts #termination],
        ),
        State([
          $QValue^Policy ( Obs_t, Act ) = sum^T_(u=t) gamma^(u-t) Rew^u $
        ], comment: [Estimate #value from rollout]),
      ),
    )
  },
  caption: [
    Computing Q-values
  ],
) <algo:q>

== Method <sec:method>
How can standard policy iteration make use of in-context learning? Policy
iteration is either model-based (@sec:model-based-planning) --- using a world-model to plan future
trajectories in the environment --- or model-free --- inferring value-estimates
without explicit planning. Both methods can be realized with in-context
learning. We choose model-based learning because planned trajectories make the
underlying logic of value-estimates explicit to our foundation model backbone,
providing a concrete instantiation of a trajectory that realizes the values.
This ties into recent work #cites(<wei_chain_2022>, <nye_show_2021>)
demonstrating that "chains of thought" can significantly improve few-shot
performance of foundation models.

Model-based RL requires two ingredients, a rollout-policy used to act during
planning and a world-model used to predict future rewards, terminations, and
states. Since our approach avoids any mutation of the foundation model's
parameters (this would require gradients), we must instead induce the
rollout-policy and the world-model using in-context learning, i.e. by selecting
appropriate prompts. We induce the rollout-policy by prompting the foundation
model with trajectories drawn from the current (or recent) behavior policy
(distinct from the rollout-policy). Similarly, we induce the world-model by
prompting the foundation models with transitions drawn from the agent's history
of experience. Note that our approach assumes access to some translation between
the state-space of the environment and the medium (language, images, etc.) of
the foundation models. This explains how an algorithm might plan and estimate
values using a foundation model. It also explains how the rollout-policy
approximately tracks the behavior policy.

How does the policy improve? When acting in the environment (as opposed to
planning), we choose the action that maximizes the estimated Q-value from the
current state (see @algo:train, @line:arg-max). At time step $t$, the agent
observes the state of the environment (denoted $Obs_t$) and executes action $Act_t = arg max_(Act in Actions) QValue^(Policy_t)(Obs_t,Act)$,
where $Actions = [Actions(1),dots,Actions(n)]$
denotes the set of $n$ actions available, $Policy_t$ denotes the policy of the
agent at time step $t$, and $QValue^Policy$ denotes the Q-estimate for policy $Policy$.
Taking the greedy ($arg max$) actions with respect to $Q^(pi_t)$ implements a
new and improved policy.

=== Computing Q-values <para:q-values>
This section provides details on the prompts that we use in our computation of
Q-values (see @algo:q pseudocode & @fig:q rollout). During training, we
maintain a buffer $Buffer$ of transitions experienced by the agent. To compute
$QValue^(Policy_t)(Obs_t, Act)$ at time step $t$ in the real-world we rollout a
simulated trajectory $Obs^1=Obs_t$, $Act^1 = Act$, $Rew^1$,
$Obs^2$, $Act^2$, $Rew^2$, $dots$, $Obs^T$, $Act^T$,
$Rew^T$, $Obs^(T+1)$ by predicting, at each simulation time step $u$: reward
$Rew^u sim LLM(Buffer_Rew, Obs^u, Act^u)$; termination
$Ter^u sim LLM(Buffer_Ter, Obs^u, Act^u)$; observation
$Obs^(u+1) sim LLM(Buffer_Obs, Obs^u, Act^u)$; action
$Act^1 sim LLM(Buffer_Act, Obs^u)$. Termination
$Ter^u$ decides whether the simulated trajectory ends at step $u$.

The prompts $Buffer_Rew$, $Buffer_Ter$ contain data sampled from the replay
buffer. For each prompt, we choose some subset of replay buffer transitions,
shuffle them, convert them to text (examples are provided in table
@sec:domains-and-prompt-format) and clip the prompt at the 4000-token Codex
context limit. We use the same method for $Buffer_Act$, except that we use
random trajectory subsequences.

In order to maximize the relevance of the prompt contents to the current
inference we select transitions using the following criteria. $Buffer_Ter$
contains $(Obs_k, Act_k, Ter_k)$ tuples such that $Act_k$ equals
$Act^u$, the action for which the LLM must infer termination.
$Buffer_Rew$ contains $(Obs_k, Act_k, Rew_k)$ tuples, again constraining $Act_k = Act^u$ but
also constraining $Ter_k = Ter^k$---that the tuple corresponds to a terminal
time-step if the LLM inferred
$Ter^u =$ true, and to a non-terminal time-step if $Ter^u =$ false. For
$Buffer_Obs$, the prompt includes $(Obs_k, Act_k Obs_k+1)$ tuples with $Act_k = Act^u$ and $Ter_k = $ false
(only non-terminal states need to be modelled).

We also maintain a balance of certain kinds of transitions in the prompt. For
termination prediction, we balance terminal and non-terminal time-steps. Since
non-terminal time-steps far outnumber terminal time-steps, this eliminates a
situation wherein the randomly sampled prompt time-steps are entirely
non-terminal, all but ensuring that the LLM will predict non-termination.
Similarly, for reward prediction, we balance the number of time-steps
corresponding to each reward value stored in $Buffer$. In order to balance two
collections of unequal size, we take the smaller and duplicate randomly chosen
members until the sizes are equal.

In contrast to the other predictions, we condition the rollout policy on
trajectory subsequences, not individual time-steps. Prompting with sequences
better enables the foundation model to apprehend the logic behind a policy.
Trajectory subsequences consist of $(Obs_k, Act_k)$ pairs, randomly clipped from
the $Recency$ most recent trajectories. More recent trajectories will, in
general demonstrate higher performance, since they come from policies that have
benefited from more rounds of improvement.

In contrast to the other predictions, we condition the rollout policy on
trajectory subsequences, not individual time-steps. Prompting with sequences
better enables the foundation model to apprehend the logic behind a policy.
Trajectory subsequences consist of $(Obs_k, Act_k)$ pairs, randomly clipped from
the $Recency$ most recent trajectories. More recent trajectories will, in
general, demonstrate higher performance, since they come from policies that have
benefited from more rounds of improvement.

Finally, the Q-value estimate is simply the discounted sum of rewards for the
simulated episode. Given this description of Q-value estimation, we now return
to the concept of policy improvement.

=== Policy-Improvement
The $arg max$ (line 5 of @algo:train) drives policy improvement in Algorithm.
Critically this is not simply a one-step improvement but a mechanism that builds
improvement on top of improvement. This occurs through a cycle in which the $arg max$
improves behavior. The improved behavior is stored in the buffer $Buffer$, and
then used to condition the rollout policy. This improves the returns generated
by the LLM during planning rollouts. These improved rollouts improve the
Q-estimates for each action. Completing the cycle, this improves the actions
chosen by the $arg max$. Because this process feeds into itself, it can drive
improvement without bound until optimality is achieved.

Note that this process takes advantage of properties specific to in-context
learning. In particular, it relies on the assumption that the rollout policy,
when prompted with trajectories drawn from a mixture of policies, will
approximate something like an average of these policies. Given this assumption,
the rollout policy will improve with the improvement of the mixture of policies
from which its prompt-trajectories are drawn. This results in a kind of rapid
policy improvement that works without any use of gradients.

#figure(
  {
    tablex(
      columns: 3,
      header-rows: 1,
      auto-vlines: false,
      [*Model*],
      [*Parameters*],
      [*Training Data*],
      [GPT-J #cite(<wang_gpt-j-6b_2021>) ],
      [6 billion],
      ["The Pile" #cite(<gao_pile_2020>), an 825GB English corpus incl. Wikipedia,
        GitHub, academic pubs],
      [InCoder cite(<fried_incoder_2022>) ],
      [6.7 billion],
      [159 GB of open-source StackOverflow code],
      [OPT-30B cite(<zhang_opt_2022>) ],
      [30 billion],
      [180B tokens of predominantly English data],
      [Codex cite(<chen_evaluating_2021>) ],
      [185 billion],
      [179 GB of GitHub code],
    )
  },
  caption: [Table of models and training data.],
  supplement: "Table",
  placement: top,
) <tab:llms>

=== Prompt-Format <para:prompt-format>

The LLM cannot take
non-linguistic prompts, so our algorithm assumes access to a textual
representation of the environment---of states, actions, terminations, and
rewards---and some way to recover the original action, termination, and reward
values from their textual representation (we do not attempt to recover states).
Since our primary results use the Codex language model (see @tab:llms), we use
Python code to represent these values (examples are available in
@tab:prompt-format).

// typstfmt::off
#figure(
  {
    set par(leading: 1em)
  tablex(columns: 2, auto-vlines: false, breakable: true,
  [*Chain*], [
    `assert state == 6` _`and state != 4`_\
`state = left()
assert reward == 0
assert not done
`
  ],
[*Distractor*], [`assert state == (6, 3)` _`and state != (4, 3)`_\
`state = left()
assert reward == 0
assert not done`],
[*Maze*], [
  `assert state == C(i=2, j=1)`
    _`and state != C(i=1, j=0)`_\
`state, reward = left()
assert reward == 0
assert not done
`
],
[*Mini Catch*], [
`assert paddle == C(2, 0)`\
_`  and ball == C(0, 4)
  and paddle.x == 2 and ball.x == 0
  and paddle.x > ball.x
  and ball.y == 4`_\
`reward = paddle.left()
ball.descend()
assert reward == 0
assert not done`
], [*Mini Invaders*], [
  `assert ship == C(2, 0) and aliens == [C(3, 5), C(1, 5)]`\
_`  and (ship.x, aliens[0].x, aliens[1].x) == (2, 3, 1)
  and ship.x < aliens[0].x
  and ship.x > aliens[1].x`_\
`ship.left()
assert reward == 0
for a in aliens:
   a.descend()
assert not done`
], [*Point-Mass*], [
`assert pos == -3.45 and vel == 0.00` _`and pos < -2 and vel == 0`_\
`pos, vel = decel(pos, vel)
assert reward == 0
assert not done`
 ]
   )},
  caption: [
    This table provides example prompts for each domain, showcasing the text format
    and hints. Hints are in italics.
  ],
)<tab:prompt-format>
// typstfmt::on

In our experiments, we discovered that the LLM world-model was unable to
reliably predict rewards, terminations, and next-states on some of the more
difficult environments. We experimented with providing domain emphhints in the
form of prompt formats that make explicit useful information --- similar to
Chain of Thought Prompting <wei_chain_2022>. For example, for the emphchain
domain, the hint includes an explicit comparison (`==` or `!=`) of the current
state with the goal state. Note that while hints are provided in the initial
context, the LLM must infer the hint content in rollouts generated from this
context.

We use a consistent idiom for rewards and terminations, namely `assert reward ==
x` and `assert done` or `assert not done`. Some decisions had to be made when
representing states and actions. In general, we strove to use simple, idiomatic,
concise Python. On the more challenging environments, we did search over several
options for the choice of hint. For examples, see @tab:prompt-format. We
anticipate that in the future, stronger foundation models will be increasingly
robust to these decisions.

== Experiments <sec:experiments>
We have three main goals in our experiments: (1) Demonstrate that the agent
algorithm can in fact quickly learn good policies, using pretrained LLMs, in a
set of six simple illustrative domains of increasing challenge; (2) provide
evidence through an ablation that the policy-improvement step --- taking the
$arg max$ over Q-values computed through LLM rollouts --- accelerates learning;
and (3) investigate the impact of using different LLMs (see
@tab:llms) --- different sizes and trained on different data, in particular,
trained on (mostly) natural language program code (Codex
and InCoder). We next describe the six domains and their associated prompt
formats, and then describe the experimental methodology and results.

=== Domains and prompt format <sec:domains-and-prompt-format>

*Chain:*
In this environment, the agent occupies an 8-state chain. The agent has three
actions: `Left`, `right`, and `try goal`. The `try goal` action always
terminates the episode, conferring a reward of 1 on state 4 (the goal state) and
0 on all other states. Because this environment has simpler transitions than
the other two, we see the clearest evidence of learning here. Note that the
initial batch of successful trajectories collected from random behavior will
usually be suboptimal, moving inefficiently toward the goal state. We include a
discount value of 0.8 in our diagram to show the improvement in efficiency of
the policy learned by the agent over the course of training. `Prompt format.`
Episodes also terminate after 8 time-steps. States are represented as numbers
from 0 to 7, as in `assert state == n`, with the appropriate integer substituted
for `n`. The actions are represented as functions `left()`, `right()`, and
`try_goal()`. For the hint, we simply indicate whether or not the current state
matches the goal state, 4.

*Distractor Chain:*
This environment is an 8-state chain, identical to the _chain_ environment,
except that the observation is a _pair_ of integers, the first indicating the
true state of the agent and the second acting as a distractor which transitions
randomly within ${0, dots, 7}$. The agent must therefore learn to ignore the
distractor integer and base its inferrences on the information contained in the
first integer. Aside from the addition of this distractor integer to the
observation, all text representations and hints are identical to the _chain_ environment.

*Maze:*
The agent navigates a small $3 times 3$ gridworld with obstacles. The agent can
move `up`, `down`, `left`, or `right`. The episode terminates with a reward of 1
once the agent navigates to the goal grid, or with a reward of 0 after 8
time-steps. This environment tests our algorithm's capacity to handle
2-dimensional movement and obstacles, as well as a 4-action state-space. We
represent the states as namedtuples --- `C(x, y)`, with integers substituted for
`x` and `y`. Similar to _chain_, the hint indicates whether or not the state
corresponds to the goal state.

*Mini Catch:*
The agent operates a paddle to catch a falling ball. The ball falls from a
height of 5 units, descending one unit per time step. The paddle can `stay` in
place (not move), or move `left` or `right` along the bottom of the 4-unit wide
plane. The agent receives a reward of 1 for catching the ball and 0 for other
time-steps. The episode ends when the ball's height reaches the paddle
regardless of whether or not the paddle catches the ball. We chose this
environment specifically to challenge the action-inference/rollout-policy
component of our algorithm. Specifically, note that the success condition in
Mini Catch allows the paddle to meander before moving under the ball---as long
as it gets there on the final time-step. Successful trajectories that include
movement _away_ from the ball thus making good rollout policies more challenging
to learn (i.e., elicit from the LLM via prompts). Again, we represent both the
paddle and the ball as namedtuples `C(x, y)` and we represent actions as methods
of the `paddle` object: `paddle.stay()`, `paddle.left()`, and `paddle.right()`.
For the hint, we call out the location of the paddle's $x$-position, the ball's $x$-position,
the relation between these positions (which is larger than which, or whether
they are equal) and the ball's $y$-position. @tab:prompt-format in the appendix
provides an example. We also include the text `ball.descend()` to account for
the change in the ball's position between states.

*Mini Invaders:*
The agent operates a ship that shoots down aliens which descend from the top of
the screen. At the beginning of an episode, two aliens spawn at a random
location in two of four columns. The episode terminates when an alien reaches
the ground (resulting in 0 reward) or when the ship shoots down both aliens (the
agent receives 1 reward per alien). The agent can move `left`, `right`, or
`shoot`. This domain highlights ICPI's capacity to learn incrementally, rather
than discovering an optimal policy through random exploration and then imitating
that policy, which is how our "No ArgMax" baseline learns (see @para:baselines).
ICPI initially learns to shoot down one alien, and then builds on this good but
suboptimal policy to discover the better policy of shooting down both aliens. In
contrast, random exploration takes much longer to discover the optimal policy
and the "No ArgMax" baseline has only experienced one or two successful
trajectories by the end of training.

We represent the ship by its namedtuple coordinate (`C(x, y)`) and the aliens as
a list of these namedtuples. When an alien is shot down, we substitute `None`
for the tuple, as in `aliens == [C(x, y), None]`. We add the text: `for a in aliens: a.descend()`, // typstfmt::off
in order to account for the change in the alien's position
between states.

*Point-Mass:*
A point-mass spawns at a random position on a
continuous line between $-6$ and $+6$ with a velocity of 0. The agent can either
`accelerate` the point-mass (increase velocity by 1) or `decelerate`
it (decrease the velocity by 1). The point-mass position changes by the amount
of its velocity each timestep. The episode terminates with a reward of 1
once the point-mass is between $-2$ and $+2$ and its velocity is 0 once again.
The episode also terminates after 8 time-steps. This domain tests the
algorithm's ability to handle continuous states.

States are represented as `assert pos == p and vel == v`, substituting
floats rounded to two decimals for `p` and `v`. The actions
are `accel(pos, vel)` and `decel(pos, vel)`. The hint
indicates whether the success conditions are met, namely the relationship
of `pos` to $-2$ and $+2$ and whether or not `vel == 0`.
The hint includes identification of the aliens' and the ship's $x$-positions
as well as a comparison between them.

== Methodology and Evaluation <para:methodology>

For the results, we record the agent's regret over the course of training
relative to an optimal policy computed with a discount factor of 0.8. For all
experiments $Recency = 8$ (the number of most recent successful
trajectories to include in the prompt). We did not have time for hyperparameter
search and chose this number based on intuition. However, the $Recency =
  16$ baseline demonstrates results when this hyperparameter is doubled. All
results use 4 seeds.

For both versions of Codex, we used the OpenAI Beta under the API Terms of Use.
For GPT-J~#cite(<wang_gpt-j-6b_2021>) , InCoder~#cite(<fried_incoder_2022>) and
OPT-30B~#cite(<zhang_opt_2022>), we used the open-source implementations from
Huggingface Transformers #cite(<wolf_transformers_2020>), each running on one
Nvidia A40 GPU. All language models use a sampling temperature of 0.1. Code for our
implementation is available at https://github.com/ethanabrooks/icpi.

#figure(
  image("figures/policy-iteration/algorithm.png"),
  caption: [
    Comparison of ICPI with three baselines, "No ArgMax," "Tabular Q," and "Nearest
    Neighbor." The $y$-axis depicts regret (normalized between 0 and 1), computed
    relative to an optimal return with a discount-factor of 0.8. The $x$-axis
    depicts time-steps during training. Error bars are standard errors from four
    seeds.
  ],
  placement: top
) <fig:algorithms>

=== Comparison of ICPI with baseline algorithms. <para:baselines>
  We compare ICPI with three
baselines (@fig:algorithms).

*The "No ArgMax" baseline* learns a good policy through random exploration and
then imitates this policy. This baseline assumes access to a "success threshold"
for each domain --- an undiscounted cumulative return greater than which a
trajectory is considered successful. The action selection mechanism emulates
ICPI's rollout policy: prompting the LLM with a set of trajectories and
eliciting an action as output. For this baseline, we only include trajectories
in the prompt whose cumulative return exceeds the success threshold. Thus the
policy improves as the number of successful trajectories in the prompt increases
over time. Note that at the start of learning, the agent will have experienced
too few successful trajectories to effectively populate the policy prompt. In
order to facilitate exploration, we act randomly until the agent experiences 3
successes.

*"Tabular Q"* is a standard tabular Q-learning algorithm, which uses a learning rate of $1.0$ and optimistically initializes the Q-values to $1.0$.

*"Matching Model"* is a baseline which uses the trajectory history instead of an
LLM to perform modelling. This baseline searches the trajectory buffer for the
most recent instance of the current state, and in the case of
transition/reward/termination prediction, the current action. If a match is
found, the model outputs the historical value (e.g. the reward associated with
the state-action pair found in the buffer). If no match is found, the modelling
rollout is terminated. Recall that ICPI breaks ties randomly during action
selection so this will often lead to random action selection.

As our results demonstrate, only ICPI learns good policies on all
domains. We attribute this advantage to ICPI's ability to generalize
from its context to unseen states and state/action pairs (unlike "Tabular Q" and
"Matching Model"). Unlike "No ArgMax," ICPI is able to learn progressively,
improving the policy before experiencing good trajectories.

#figure(
  image("figures/policy-iteration/ablation.png"),
  caption: [Comparison of ICPI with ablations. The $y$-axis depicts
    regret (normalized between 0 and 1), computed relative to an optimal
    return with a discount-factor of 0.8. The $x$-axis depicts time-steps
    during training. Error bars are standard errors from four seeds.],
) <fig:ablations>

=== Ablation of ICPI components

With these experiments, we
ablate those components of the algorithm which are not, in principle, essential
to learning (@fig:ablations). "No Hints" ablates the hints described in
the <para:prompt-format> paragraph. "No Balance" removes the balancing
of different kinds of time-steps described in the <para:q-values>
paragraph (for example, $Buffer_Ter$ is allowed to contain an unequal number
of terminal and non-terminal time-steps). The "No Constraints" baseline removes
the constraints on these time-steps described in the same paragraph. For
example,
$Buffer_Rew$ is allowed to contain a mixture of terminal and non-terminal
time-steps (regardless of the model's termination prediction). Finally, "$Recency=16$" baseline prompts the rollout
policy with the last 16 trajectories (instead of the last 8, as in
ICPI). We find that while some ablations match ICPI's
performance in several domains, none match its performance on all six.


=== Comparison of Different Language Models
 While our lab lacks the
resources to do a full study of scaling properties, we did compare several
language models of varying size (see @fig:language-models). See @tab:llms for details about
these models. Both `code-davinci-002` and `code-cushman-001` are
variations of the Codex language model. The exact number of parameters in these
models is proprietary according to OpenAI, but #cite(<chen_evaluating_2021>)
describes Codex as fine-tuned from GPT-3 #cite(<brown_language_2020>), which
contains 185 billion parameters. As for the distinction between the variations,
the OpenAI website describes `code-cushman-001` as "almost as capable as
Davinci Codex, but slightly faster."

We found that the rate of learning and final performance of the smaller models fell significantly short of Codex on all but the simplest domain, _chain_.
Examining the trajectories generated by agents trained using these models, we
noted that in several cases, they seemed to struggle to apprehend the
underlying "logic" of successful trajectories, which hampered the ability
of the rollout policy to produce good actions.
Since these smaller models were not trained on identical data, we are unable to
isolate the role of size in these results. However, the failure of all of these
smaller models to learn suggests that size has some role to play in performance.
We conjecture that larger models developed in the future may demonstrate
comparable improvements in performance over our Codex model.

=== Limitations <para:limitations>
ICPI can theoretically work on any control task with discrete actions, due to the
guarantees associated with policy iteration.
However, since our implementation uses Codex, the domains in our paper were limited by the ability to encode states as text and to fit these encodings in the model's context window. Moreover, Codex demonstrated a limited ability to
predict transitions and actions in more complex domains. As sequence models
mature, we anticipate that more domains will become tractable for ICPI. We also
note that reliance on the proprietary OpenAPI API limits exact reproduction of these
results.


=== Societal Impacts <para:impacts>
An extensive literature #cites(<tamkin2021understanding>,<abid2021persistent>,<liang2021towards>,<pan2023risk>)
has explored the possible positive and negative impacts of LLMs. Some of this
work has explored mitigation strategies. In extending LLMs to RL, our work
inherits these benefits and challenges. We highlight two concerns: the use of
LLMs to spread misinformation and the detrimental carbon cost of training and
using these models.

#figure(
  image("figures/policy-iteration/language-model.png"),
  caption: [Comparison of different language models used to implement ICPI. The $y$-axis depicts
    regret (normalized between 0 and 1), computed relative to an optimal
    return with a discount-factor of 0.8. The $x$-axis depicts time-steps of
    training. Error bars are standard errors from four seeds.],
) <fig:language-models>


== Conclusion
Our main contribution in this chapter is a method for implementing policy iteration algorithm
using Large Language Models and the mechanism of in-context learning. The
algorithm uses a foundation model as both a world model and policy to compute
Q-values via rollouts. Although we presented the method here as text-based, it
is general enough to be applied to any foundation model that works through
prompting, including multi-modal models like #cite(<reed_generalist_2022>) and
#cite(<seo_harp_2022>). In experiments we showed that the algorithm works in six
illustrative domains imposing different challenges for ICPI, confirming the
benefit of the LLM-rollout-based policy improvement. While the empirical results
are preliminary, we believe the approach provides an important new way to use
LLMs that will increase in effectiveness as the models themselves become more
powerful.



// #bibliography("main.bib", style: "american-society-of-civil-engineers")