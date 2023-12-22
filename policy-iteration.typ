#import "math.typ": *
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms

#show: show-algorithms

#set heading(numbering: "1.1")

= In-Context Policy Iteration
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
#cites(<lu_pretrained_2021>, <brown_language_2020>, <chan_data_2022>).
#cites(<brown_language_2020>, <wei_emergent_2022>). A litany of existing work
has explored methods for applying this remarkable capability to downstream tasks
(see @sec:related), including Reinforcement Learning (RL). Most work in this
area either (1) assumes access to expert demonstrations---collected either from
human experts
#cites(<huang_language_2022>, <baker_video_2022>), or domain-specific
pre-trained RL agents
#cites(
  <chen_decision_2021>,
  <lee_multi-game_2022>,
  <janner_offline_2021>,
  <reed_generalist_2022>,
  <xu_prompting_2022>,
).---or (2) relies on gradient-based methods---e.g. fine-tuning of the
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
tasks---_chain, distractor-chain, maze, mini-catch, mini-invaders_, and
_point-mass_---in which the algorithm very quickly finds good policies. We also
compare five pretrained Large Language Models (LLMs), including two different
size models trained on natural language---OPT-30B and GPT-J---and three
different sizes of a model trained on program code---two sizes of Codex as well
as InCoder. On our six domains, we find that only the largest model (the
`code-davinci-001` variant of Codex) consistently demonstrates learning.

#figure(
  image("figures/policy-iteration/action-selection.svg"),
  caption: [
    For each possible action $Actions(1), dots.h, Actions(n)$, the LLM generates a
    rollout by alternately predicting transitions and selecting actions. Q-value
    estimates are discounted sums of rewards. The action is chosen greedily with
    respect to Q-values. Both state/reward prediction and next action selection use
    trajectories from // $\Buffer$
    to create prompts for the LLM. Changes to the content of // $\Buffer$
    change the prompts that the LLM receives, allowing the model to improve its
    behavior over time.
  ],
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
them with a foundation model, model representations as input to deep RL
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
example, #cite(<huang_language_2022>) use a frozen LLM as a planner for everyday
household tasks by constructing a prefix from human-generated task instructions,
and then using the LLM to generate instructions for new tasks. This work is
extended by #cite(<huang_inner_2022>). Similarly,
#cite(<ahn_as_2022>) use a value function that is trained on human
demonstrations to rank candidate actions produced by an LLM.
#cite(<baker_video_2022>) use human demonstrations to train the foundation model
itself: they use video recordings of human Minecraft players to train a
foundation models that plays Minecraft. Works that rely on pretrained RL agents
include #cite(<janner_offline_2021>) who train a "Trajectory Transformer" to
predict trajectory sequences in continuous control tasks by using trajectories
generated by pretrained agents, and #cite(<chen_decision_2021>), who use a
dataset of offline trajectories to train a "Decision Transformer" that predicts
actions from state-action-reward sequences in RL environments like Atari. Two
approaches build on this method to improve generalization:
#cite(<lee_multi-game_2022>) use trajectories generated by a DQN agent to train
a single Decision Transformer that can play many Atari games, and
#cite(<xu_prompting_2022>) use a combination of human and artificial
trajectories to train a Decision Transformer that achieves few-shot
generalization on continuous control tasks. #cite(<reed_generalist_2022>) take
task-generality a step farther and use datasets generated by pretrained agents
to train a multi-modal agent that performs a wide array of RL (e.g. Atari,
continuous control) and non-RL (e.g. image captioning, chat) tasks.

Some of the above works include non-expert demonstrations as well.
#cite(<chen_decision_2021>) include experiments with trajectories generated by
random (as opposed to expert) policies. #cite(<lee_multi-game_2022>) and
#cite(<xu_prompting_2022>) also use datasets that include trajectories generated
by partially trained agents in addition to fully trained agents. Like these
works, our proposed method (ICPI) does not rely on expert demonstrations---but
we note two key differences between our approach and existing approaches.
Firstly, ICPI only consumes self-generated trajectories, so it does not require
any demonstrations (like #cite(<chen_decision_2021>) with random trajectories,
but unlike #cite(<lee_multi-game_2022>), #cite(<xu_prompting_2022>), and the
other approaches reviewed above). Secondly, ICPI relies primarily on in-context
learning rather than in-weights learning to achieve generalization (like
#cite(<xu_prompting_2022>), but unlike #cite(<chen_decision_2021>) \&
#cite(<lee_multi-game_2022>)). For discussion about in-weights vs. in-context
learning see #cite(<chan_data_2022>).

=== Gradient-based training \& finetuning on RL tasks
Many approaches that use foundation models for RL involve specifically training
or fine-tuning on RL tasks. For example,

#cites(
  <janner_offline_2021>,
  <chen_decision_2021>,
  <lee_multi-game_2022>,
  <xu_prompting_2022>,
  <baker_video_2022>,
  <reed_generalist_2022>,
)
all use models that are trained from scratch on tasks of interest, and
#cites(<singh_know_2022>, <ahn_as_2022>, <huang_inner_2022>) combine frozen
foundation models with trainable components or adapters. In contrast,
#cite(<huang_language_2022>) use frozen foundation models for planning, without
training or fine-tuning on RL tasks. Like #cite(<huang_language_2022>), ICPI
does not update the parameters of the foundation model, but relies on the frozen
model's in-context learning abilities. However, ICPI gradually builds and
improves the prompts within the space defined by the given fixed text-format for
observations, actions, and rewards (in contrast to #cite(<huang_language_2022>),
which uses the frozen model to select good prompts from a given fixed library of
goal/plan descriptions).

=== In-Context learning
Several recent papers have specifically studied in-context learning.
#cite(<laskin2022context>) demonstrates an approach to performing in-context
reinforcement learning by training a model on complete RL learning histories,
demonstrating that the model actually distills the improvement operator of the
source algorithm. % #cite(<min_rethinking_2022>) demonstrates that LLMs can
learn in-context, even % when the labels in the prompt are randomized,
problemetizing the conventional % understanding of in-context learning and
showing that label distribution is more % important than label correctness.
#cite(<chan_data_2022>) and #cite(<garg_what_2022>) provide analyses of the
properties that drive in-context learning, the first in the context of image
classification, the second in the context of regression onto a continuous
function. These papers identify various properties, including "burstiness,"
model-size, and model-architecture, that in-context learning depends on.
#cite(<chen_relation_2022>) studies the sensitivity of in-context learning to
small perturbations of the context. They propose a novel method that uses
sensitivity as a proxy for model certainty. Some recent work has explored
iterative forms of in-context learning, similar to our own. For example,
#cite(<shinn2023reflexion>) and #cite(<madaan2023self>) use iterative
self-refinement to improve the outputs of a large language model in a natural
language context. These approaches rely on the ability of the model to examine
and critique its own outputs, rather than using policy iteration as our method
does.

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
            $formatAction(Act) gets arg max_Act QValue^Policy (Obs_t, Act) $,
            comment: [ Choose #formatAction("action") with highest #formatValue("value") ],
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
iteration is either _model-based_---using a world-model to plan future
trajectories in the environment---or _model-free_---inferring value-estimates
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
current state (see @algo:train pseudocode, line 6). At time step $t$, the agent
observes the state of the environment (denoted $Obs_t$) and executes action $Act_t = arg max_(Act in Actions) QValue^(Policy_t)(Obs_t,Act)$,
where $Actions = [1/2,Actions(1),dots,Actions(n)]$
denotes the set of $n$ actions available, $Policy_t$ denotes the policy of the
agent at time step $t$, and $QValue^Policy$ denotes the Q-estimate for policy $Policy$.
Taking the greedy ($arg max$) actions with respect to $Q^(pi_t)$ implements a
new and improved policy.

=== Computing Q-values <para:q-values>
This section provides details on the prompts that we use in our computation of
Q-values (see @algo:q pseudocode & Figure @fig:q rollout). During training, we
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
general demonstrate higher performance, since they come from policies that have
benefited from more rounds of improvement.

Finally, the Q-value estimate is simply the discounted sum of rewards for the
simulated episode. Given this description of Q-value estimation, we now return
to the concept of policy improvement.

=== Domains and prompt format <sec:domains-and-prompt-format>

#bibliography("main.bib", style: "association-for-computing-machinery")