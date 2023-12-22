#import "math.typ": *
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms

#show: show-algorithms

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
area either (1) assumes access to expert demonstrations --- collected either
from human experts
#cites(<huang_language_2022>, <baker_video_2022>), or domain-specific
pre-trained RL agents
#cites(
  <chen_decision_2021>,
  <lee_multi-game_2022>,
  <janner_offline_2021>,
  <reed_generalist_2022>,
  <xu_prompting_2022>,
). --- or (2) relies on gradient-based methods --- e.g. fine-tuning of the
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

We illustrate the algorithm empirically on six small illustrative RL tasks---
_chain, distractor-chain, maze, mini-catch, mini-invaders_, and
_point-mass_---in which the algorithm very quickly finds good policies. We also
compare five pretrained Large Language Models (LLMs), including two different
size models trained on natural language---OPT-30B and GPT-J---and three
different sizes of a model trained on program code---two sizes of Codex as well
as InCoder. On our six domains, we find that only the largest model (the
`code-davinci-001` variant of Codex) consistently demonstrates learning.

#figure(
  image("figures/policy-iteration/action-selection-fig-final.svg"),
  caption: [
    For each possible action $Action(1), dots.h, Action(n)$, the LLM generates a
    rollout by alternately predicting transitions and selecting actions. Q-value
    estimates are discounted sums of rewards. The action is chosen greedily with
    respect to Q-values. Both state/reward prediction and next action selection use
    trajectories from // $\Buffer$
    to create prompts for the LLM. Changes to the content of // $\Buffer$
    change the prompts that the LLM receives, allowing the model to improve its
    behavior over time.
  ],
) <fig:q-rollouts>

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
Finally, a recent set of approaches treat RL as a sequence modeling problem and
use the foundation models itself to predict states or actions. This section will
focus on this last category.

=== Learning from demonstrations
Many recent sequence-based approaches to reinforcement learning use
demonstrations that come either from human experts or pretrained RL agents. For
example, #cite(<huang_language_2022>) use a frozen LLM as a planner for everyday
household tasks by constructing a prefix from human-generated task instructions,
and then using the LLM to generate instructions for new tasks. This work is
extended by #cite(<huang_inner_2022>). Similarly, #cite(<ahn_as_2022>) use a
value function that is trained on human demonstrations to rank candidate actions
produced by an LLM. #cite(<baker_video_2022>) use human demonstrations to train
the foundation model itself: they use video recordings of human Minecraft
players to train a foundation models that plays Minecraft. Works that rely on
pretrained RL agents include #cite(<janner_offline_2021>) who train a
``Trajectory Transformer'' to predict trajectory sequences in continuous control
tasks by using trajectories generated by pretrained agents, and
#cite(<chen_decision_2021>), who use a dataset of offline trajectories to train
a ``Decision Transformer'' that predicts actions from state-action-reward
sequences in RL environments like Atari. Two approaches build on this method to
improve generalization: #cite(<lee_multi-game_2022>) use trajectories generated
by a DQN agent to train a single Decision Transformer that can play many Atari
games, and #cite(<xu_prompting_2022>) use a combination of human and artificial
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
learning see #cite(<chan_data_2022>). % see the next section.

=== Gradient-based Training \& Finetuning on RL Tasks
Most approaches involve training or fine-tuning foundation models on RL tasks.
For example,
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

=== In-Context Learning
Several recent papers have specifically studied in-context learning. #cite(<min_rethinking_2022>) demonstrates
that LLMs can learn in-context, even when the labels in the prompt are
randomized, problemetizing the conventional understanding of in-context learning
and showing that label distribution is more important than label correctness. #cite(<chan_data_2022>) and
#cite(<garg_what_2022>) provide analyses of the properties that drive in-context
learning, the first in the context of image classification, the second in the
context of regression onto a continuous function. These papers identify various
properties, including "burstiness," model-size, and model-architecture, that
in-context learning depends on. #cite(<chen_relation_2022>) studies the
sensitivity of in-context learning to small perturbations of the context. They
propose a novel method that uses sensitivity as a proxy for model certainty.

#let formatState = (s) => text(fill: green)[*#s*]
#let formatAction = (s) => text(fill: blue)[*#s*]
#let formatValue = (s) => text(fill: purple)[*#s*]
#let state = formatState("state")
#let action = formatAction("action")
#let value = formatValue("value")

#show figure.where(kind: "algorithm"): it => {
  let booktabbed = block(stroke: (y: 1.3pt), inset: 0pt, breakable: true, width: 100%, {
    set align(left)
    block(inset: (y: 5pt), width: 100%, stroke: (bottom: .8pt), {
      strong({
        it.supplement
        sym.space.nobreak
        counter(figure.where(kind: "lovelace")).display(it.numbering)
        [: ]
      })
      it.caption
    })
    block(inset: (bottom: 5pt), breakable: true, it.body)
  })
  let centered = pad(x: 5%, it)
  set align(left)
  if it.placement in (auto, top, bottom) {
    place(it.placement, float: true, centered)
  } else {
    centered
  }
}

#algorithm-figure({
  import "algorithmic.typ": *
  algorithm(..
  For(
    "each step in episode",
    State([ Observe #state ]),
    ..
    For(
      [each #action in action space],
      State([ Compute #value given #state and #action ]),
    ),
    State([
      Choose #action with highest #value
    ]),
    State([ Receive reward and next #state ]),
    State([ Add interaction to replay buffer ]),
  ))
}, caption: [
  ICPI
])