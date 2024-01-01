#import "math.typ": *
#import "@preview/cetz:0.1.2": canvas, draw, tree
#import "algorithmic.typ": algorithm-figure, show-algorithms, alg-counter

#alg-counter.step() // deal with dumb counter bug

= Bellman Update Networks

In the previous two chapters, we demonstrated the capacity of a sequence model
to implement policy iteration in-context, enabling fast adaptation to novel RL
settings without recourse to gradients. This approach relies on some mechanism
for estimating state-action values, with respect to which the policy can choose
actions greedily.

Previously, we satisfied this requirement by using a model to perform simulated
rollouts and using these rollouts to make monte-carlo estimates. This approach
has two drawbacks. First, monte-carlo estimates are generally susceptible to
high variance. Second and perhaps more fundamentally, our previous methods
focused on training an accurate model when value accuracy — not model accuracy —
was our ultimate concern. As a result, our model learned to focus equally on all
parts of the observation rather than skewing its resources toward those parts
that contributed to the expected value #cite(<grimm2020value>) and away from
parts that are purely "decorative." Conversely, error in modeling any part of
the observation, decorative or otherwise, could throw off the value estimate: a
model trained on inputs containing only ground-truth observations might fail to
generalize to observations corrupted by modeling error.

In this work, we attempt to address these issues by proposing a method for
estimating value directly instead of using model-based rollouts. In the next
section we will introduce the motivation, concept, and implementation for
Bellman Update Networks. In the subsequent section, we will review some results
comparing this approach with some baselines.

== Preliminaries

=== Review of In-Context Model-Based Planning <review-of-in-context-model-based-planning>
In the preceding chapter, we described work in which we trained a causal model
to map a temporally sequential history of observations $Obs_(<= t)$, actions
$Act_(<= t)$, and rewards $Rew_(<t)$, to predictions of the next observation $Obs_(t+1)$ and
next reward $Rew_(t)$. Our model optimized the following loss:

$ Loss^"AD"_theta := -E_(History_t^n)[ sum_(t=1)^(T-1) log Prob_theta (Act^n_t |
History^n_t) + log Prob_theta (Rew^n_t, Ter^n_t, Obs^n_(t+1) | History^n_t,
Act^n_t)] $

Here, $History^n_t$ is a trajectory drawn from the $n^"th"$ task/policy pair:
$ History^n_t := (Act^n_(t-1-Recency),
Rew^n_(t-1-Recency), Ter^n_(t-1-Recency), Obs^n_(t-Recency), dots,
Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t) $ <eq:history-adpp>
Henceforth in this chapter, we omit the $n$ superscript for the sake of brevity
and assume that there is one task and policy per trajectory. In the preceding
chapter, we implemented our model as a causal transformer
#cite(<vaswani2017attention>). Because the model was conditioned on history, it
demonstrated the capability to adapt in-context to settings with novel
transition or reward functions.

=== Naive Alternative Method <sec:naive>

An almost identical technique might be used to predict value functions in place
of next-observations and next-rewards. In principle, a history of observations,
actions, and rewards should be sufficient to infer the dynamics and reward of
the environment as well as the current policy, all that is necessary to estimate
value. Our model would then minimize the following loss:

$ Loss_theta &:= -E_History_t [sum_(t=1)^(T-1)sum_(Act in Actions) log Prob_theta (Highlight(QValue (Obs_t, Act)) |
History_t)]
$ <eq:loss1>

where $Obs_t$ is the last observation in $History_t$ and $QValue (Obs_t, Act)$ is
the value of observation $Obs_t$ and action $Act$. We note that in general,
ground-truth targets are not available for such a loss. However, they may be
approximated using bootstrapping methods as in #cite(<mnih2015human>, form: "prose")
or #cite(<kumar2020conservative>, form: "prose").

One question that we seek to understand is the extent to which this approach
generalizes to novel tasks and policies. We observe that the mapping from a
history of observations, actions, and rewards to values is non-trivial,
requiring the model to infer the policy and the dynamics of the environment, and
to implicitly forecast these estimates for multiple time steps. As a result, it
is reasonable to anticipate some degree of memorization.

== Proposed Method <sec:bellman-network-method>

In the case of context-conditional models such as transformers, one factor that
has a significant impact on memorization is the extent to which the model
attends to information in its context --- "in-context" learning --- as opposed
to the information distilled in its weights during training --- "in-weights"
learning. The former will be sensitive to new information introduced during
evaluation while the latter will not. As #cite(<chan2022data>, form: "prose") suggests,
the relevance or predictive power of information in a model's context strongly
influences the balance of in-weights vs. in-context learning.

With this in mind, we note that the ground-truth values associated with each
time-step in the model's context would be highly predictive of the value of the
current state. A model provided this information in its context should attend
more strongly to its context and memorize less. This would entail the following
redefinition of $History_t$, the history on which we condition the model's
predictions:

$ History_t := (Act_(t-1-Recency),
Rew_(t-1-Recency), Ter_(t-1-Recency), Obs_(t-1-Recency), Highlight(QValue (Obs_(t-Recency+1), dot.c)) dots,
Highlight(QValue (Obs_t, dot.c)),
Act_(t-1), Rew_(t-1), Ter_(t-1), Obs_t, ) $ <eq:history-bellman-network>

The question remains how such input values might be obtained. After all, in a
setting where we intend to estimate values, we should not assume that we already
have access to good value estimates! To address this chicken-and-egg dilemma, we
propose an alternative approach to predicting values _directly_ and instead take
inspiration from the classic Policy Evaluation algorithm #cite(<sutton2018reinforcement>),
which iteratively improves a value estimate using the Bellman update equation:

$ QValue_Highlight(k) (Obs_t, Act_t) := Rew(Obs_t, Act_t) + gamma E_(Obs_(t+1) Act_(t+1))
[ QValue_Highlight(k-1) (Obs_(t+1), Act_(t+1)) ] $

For any $QValue_0$ and for sufficiently large $k$, this equation converges to
the true value of $QValue$. We incorporate a similar approach in our method,
proposing the following loss function:

$ Loss^"BUN"_theta &:= -E[ sum_(t=1)^(T-1)sum_(Act in Actions) log Prob_theta (QValue_Highlight(k) (Obs_t, Act) |
History^(QValue_Highlight(k-1))_(t))]
\
#text("where")
History^(QValue_Highlight(k-1))_(t) &:= (Act_(t-1-Recency),

Rew_(t-1-Recency), Ter_(t-1-Recency), Obs_(t-Recency), QValue_Highlight(k-1) (Obs_(t-Recency), dot.c) dots,
QValue_Highlight(k-1) (Obs_(t-1), dot.c),
Act_(t-1), Rew_(t-1), Ter_(t-1), Obs_t, )
\
$ <eq:loss>

We call a model $P_theta$ that minimizes this loss a Bellman Update Network
(BUN). Initially, we may set $QValue_0$ to any value. By feeding this initial
input into the network and then auto-regressively feeding the outputs back in,
we may obtain an estimate of both the target value, $QValue_(k) (Obs_t, Act)$ and
the context values $QValue_(k-1) (Obs_(t-Recency+1), dot.c),..., QValue_(k-1) (Obs_(t-1), dot.c)$.
By conditioning predictions of $QValue_k$ on a context containing estimates of $QValue_(k-1)$,
we gain many of the benefits of the context proposed in
@eq:history-bellman-network, in terms of promoting in-context learning over
in-weights learning, without privileged access to ground-truth values.

This approach entails a tradeoff. By training on $k < infinity$, we
significantly increase the space of inputs for which the model must learn good
representations. Therefore, the method takes longer to train and demands more of
the capacity of the model. What we gain are more robust representations, capable
of better generalization to unseen settings.

=== Architecture <sec:architecture>

#figure(placement: top, {
  set text(font: "PT Sans", size: 10pt)

  canvas(length: 1cm, {
    import draw: *
    let plusOne(t) = {
      if type(t) == "integer" {
        t + 1
      } else if type(t) == "string" {
        $#t + 1$
      } else {
        type(t)
      }
    }
    let transition(t) = {
      let t1 = plusOne(t)
      (
        $Act_#t$,
        $Policy(dot.c, Obs_#t)$,
        $Rew_#t$,
        $Ter_#t$,
        $Obs_(#t1)$,
        $Value_k (Obs_#t1)$,
      )
    }
    let cell(content, .. args) = box(
      height: 1cm,
      radius: .1cm,
      stroke: black,
      ..args,
      align(center + horizon, content),
    )
    // grid((0, 0), (15, 8), stroke: (paint: gray, dash: "dotted"))
    content((7.0, 1.5), $dots.c$)
    content((7.0, 4.5), $dots.c$)
    for (t, start) in ((0, 0), ("t", 9)) {
      for (i, component) in transition(t).enumerate(start: start) {
        content((i, .25), $component$)
        line((i, .5), (i, 1), mark: (end: ">"))
      }
      line((start + 2.5, 2), (start + 2.5, 2.5), mark: (end: ">"))
      content((start + 2.5, 1.5), cell(width: 6cm, fill: red, "GRU"))
      line((start + 2.5, 3.5), (start + 2.5, 4), mark: (end: ">"))
      content((start + 2.5, 4.5), $Value_(k+1)(Obs_#plusOne(t))$)
    }
    content((7, 3), cell(width: 15cm, fill: orange, [Transformer]))
  })
}, caption: [Architecture diagram for the Bellman Update Network.])<fig:architecture>

Before we describe the procedure for training the Bellman Update Network, we
describe the inputs that the model receives, the architectures used to encode
them, and the targets on which the outputs regress. We assume that we are given
a dataset of observations $Obs$, actions $Act$, rewards $Rew$, terminations $Ter$,
and policy logits $Policy(dot.c|Obs)$. We also assume that we have estimates of
value at different numbers of Bellman updates $k$, although we defer the
explanation of how to acquire these to @sec:train-bellman-network.

As @fig:architecture illustrates, the inputs to the network are a sequence of
transitions from the dataset. In principle, these transitions need not be
chronological, except in a partially observed setting. Importantly, the
observations are offset by one index from the rest of the components of the
transition. This will be explained in @sec:implementation. Each transition gets
encoded and summarized into a single fixed-size vector. We compared several
methods for doing this, including small positional transformers, and found that
Gated Recurrent Unit (GRU) #cite(<cho2014learning>) demonstrated the strongest
performance. Each of these transition vectors gets passed through a transformer
network (one vector per transformer index) and the final outputs are projected
to scalars.

Note that in @fig:architecture, we provide $Value_k$ and not $QValue_k$ to the
model, as in @eq:loss. We found that computing

$ Value_k (Obs_t) := sum_Act Policy QValue_k (Obs_t, Act) $ <eq:value>

and providing this as input to the network, rather than providing the full array
of Q-values, improved the speed and stability of learning.

To regress the outputs of the model onto value targets, we use mean-square-error
loss:

$ Loss_theta &:= sum_(t=1)^(T-1) [QValue_theta (Obs_t, Act_t | History^(Value_Highlight(k))_t) - QTar_Highlight(k+1)(Obs_t, Act_t)]^2 $ <eq:loss-bellman-update-network>

where $History^(Value_k)_t$ is a sequence of transitions containing values $Value_k$ and $QTar_(k+1)$ is
a target Q value computed using bootstrapping (details in
@sec:train-bellman-network). This loss is equivalent to @eq:loss when $Prob_theta$ is
normally distributed with fixed standard deviation.

=== Training procedure <sec:train-bellman-network>

Here we describe a practical procedure for computing values to serve as inputs
and targets to the network, and for training the network. For all values of $k$ greater
than 1, we must choose between using inaccurate bootstrap values or adopting a
curriculum. Each approach introduces a different form of non-stationarity into
the training procedure. We favor the latter, since it avoids training the
network on targets before they are mostly accurate.

Our curriculum initially trains
$QValue_1$ bootstrapped from $QValue_0$, which we set to *$0$*. We proceed
iteratively through higher order values until $QValue_K approx QValue_(K-1)$. At
each step in the curriculum, we continue to train $QValue_k$ for all values of $k
in 1, ..., K$ (see @line:for-k of @alg:train-bellman-network). This allows the
network to continue improving its estimates for lower values of $k$ even as it
begins to train on higher values, thereby mitigating the effects of compound
error. Another benefit of continuing to train lower values of $k$ is that these
estimates can benefit from backward transfer as the curriculum progresses. As
the network produces estimates, we use them both to train the network (see
@line:optimize in @alg:train-bellman-network) but also to produce bootstrap
targets for higher values of $k$ (see @line:bootstrap).

#let formatObs = text.with(fill: green)

#let FormatV(x) = text(fill: orange, x)
#let FormatQ(x) = text(fill: green, x)
#let FormatQTar(x) = text(fill: blue, x)
#let FormatBUN(x) = text(fill: aqua, x)
#let Value_k = FormatV($Value_k$)
#let QValue_k = FormatQ($QValue_k$)
#let QValue_theta = FormatBUN($QValue_theta$)

#algorithm-figure(
  {
    import "algorithmic.typ": *

    algorithm(
      Input($Recency, Buffer$, comment: [Context length, RL data ]),
      State(
        $FormatQ(QValue_0) gets bold(0)$,
        comment: [Initialize #FormatQ("Q-estimates") to zero.],
      ),
      State($K gets 0$),
      ..Repeat(
        ..Repeat(
          ..For(
            $k = 0, ..., K$,
            label: <line:for-k>,
            State(
              $(Act_t, Policy(dot.c|Obs_t), Rew_t, Ter_t, Obs_(t+1) )_(t=0)^Recency sim Buffer$,
              comment: "sample sequence from data.",
            ),
            ..For(
              $t = 1, ..., 1 + Recency$,
              State(
                [$#Value_k (Obs_t) gets sum_Act Policy(Act | Obs_t) #QValue_k (Obs_t, Act)$],
                comment: [Compute #FormatV("values") from #FormatQ("Q-values")],
                label: <line:vest>,
              ),
            ),
            State(
              [$History^(#Value_k) gets (Act_t, Policy(dot.c|Obs_t), Rew_t, Ter_t, Obs_(t+1), #Value_k (Obs_(t+1)) )_(t=0)^Recency$ ],
              comment: [pair transitions with #FormatV("values")],
              label: <line:pair>,
            ),
            ..For(
              $t=1, ..., 1+Recency$,
              State(
                $FormatQTar(QTar_(k+1)) (Obs_t, Act_t) gets Rew_t + (1-Ter_t) gamma #Value_k (Obs_(t+1))$,
                comment: [Bootstrap #FormatQTar("target") for observed actions.],
                label: <line:bootstrap>,
              ),
            ),
            State(
              [$FormatQ(QValue_(k+1)) gets #QValue_theta (History^(#Value_k))$],
              comment: [Use #FormatBUN("Bellman Update Network") to estimate #FormatQ("values")],
              label: <line:forward>,
            ),
            State(
              [minimize $sum_t [#QValue_theta (Obs_t, Act_t | History^(#Value_k)_t) - FormatQTar(QTar_(k+1))(Obs_t, Act_t)]^2 $],
              comment: "Optimize predictions",
              label: <line:optimize>,
            ),
          ),
          $FormatQ(QValue_(k+1)) approx FormatQTar(QTar_(k+1))$,
        ),
        State($K gets K + 1$),
        $FormatQ(QValue_(k + 1)) approx FormatQ(QValue_k)$,
      ),
    )
  },
  caption: [ Training the Bellman Update Network. ],
  placement: top,
) <alg:train-bellman-network>

=== Implementation Details <sec:implementation>
We implement the Bellman Update Network as a causal transformer, using the GPT2 #cite(<radford2019language>)
implementation from #link("https://huggingface.co/", [www.huggingface.co]). Why
is causal masking necessary, given that the target does not appear in the input
to the model? To answer this question, we must draw attention to a disparity
between the outputs from the model on @line:forward of
@alg:train-bellman-network and the targets used to train the model on
@line:optimize. For each input observation $Obs_t$, we require the model to
infer a vector of values, *$QValue(Obs_t, dot.c)$*, one for each action in the
action space. However, we are only able to train the model on the single action
observed in the dataset for that transition. If the model is able to observe
both the input observation $Obs_t$ and the action $Act_t$ on which we condition
the target value, the model will neglect all predictions besides $QValue(Obs_t, Act_t)$.
That is, it will learn good predictions of $QValue(Obs_t, Act)$ for $Act = Act_t$,
the action that appears in the dataset, but not for the other actions in the
action space. To prevent this degenerate outcome, we use masking to prevent the
model from observing $Act_t$ when conditioning on $Obs_t$. This is also why we
offset the observations and value predictions by one index, as in
@fig:architecture.

One consequence of masking is that predictions for values early in the sequence
are poor in comparison to predictions later in the sequence, since they benefit
from less context. Repeated bootstrapping from these poor predictions propagates
error throughout the sequence. To mitigate this, we rotate the sequence by
fractions, retaining predictions from only the last fraction. For example, if we
break the sequence into three equal fractions $(X_1, X_2, X_3)$, we apply three
rotations, yielding rotated sequences $(X_1, X_2, X_3)$, $(X_2, X_3, X_1)$, and $(X_3,
X_1, X_2)$. We pass each rotation through the model, and for each rotation, we
retain only the predictions for $X_3$, $X_1$, and $X_2$ respectively. We use
this rotation procedure to produce the Q estimates on @line:forward of
@alg:train-bellman-network.

#alg-counter.update(4)

Another important detail is that the bootstrap step on @line:bootstrap of
@alg:train-bellman-network leads to instability when generating targets for
lower targets of $k$ which the model has previously trained on. To mitigate
this, we interpolate $QTar_(k+1)$ with its previous value, using an
interpolation factor of $.5$.

=== Downstream Evaluation <sec:downstream>
Once the network is trained, we can use it to estimate values in a new setting
by using the iterative method we described in @sec:bellman-network-method. In
addition, we can use the estimates to act, by choosing actions greedily by value
(see @alg:eval-bellman-network).

#algorithm-figure(
  {
    import "algorithmic.typ": *
    algorithm(
      Input(
        $Recency, K, T$,
        comment: [Context length, iterations, evaluation length ],
      ),
      State(
        $FormatQ(QValue_0) gets bold(0)$,
        comment: "Initialize Q-estimates to zero.",
      ),
      State(
        [$(Obs_t, Act_t, Policy(dot.c|Obs_t), Rew_t, Ter_t)_(t=0)^Recency sim$ random
          behavior],
        comment: "Fill transformer context with random behavior.",
      ),
      State([$Obs_(Recency+1) gets$ reset environment]),
      ..For(
        $t_0=1, ..., 1+T$,
        ..For(
          $k=0,...,K$,
          ..For(
            $t = t_0, ..., t_0 + Recency$,
            State(
              [$#Value_k (Obs_t) gets sum_Act Policy(Act | Obs_t) #QValue_k (Obs_t, Act)$],
              comment: [Compute #FormatV("values") from #FormatQ("Q-values")],
            ),
          ),
          State(
            [$History^(#Value_k) gets (Act_t, Policy(dot.c|Obs_t), Rew_t, Ter_t, Obs_(t+1),#Value_k (Obs_(t+1)) )_(t=t_0)^(t_0 + Recency)$ ],
            comment: [pair transitions with #FormatV("values")],
            label: <line:pair>,
          ),
          State(
            [$FormatQ(QValue_(k+1)) gets #QValue_theta (History_Value_k)$],
            comment: [Use the #FormatBUN("Bellman Update Network") to estimate values.],
          ),
        ),
        State($t gets t_0 + Recency$),
        State(
          $Act_t gets arg max_Act FormatQ(QValue_(K+1)) (Obs_t, Act)$,
          comment: "Choose the action with the highest value.",
        ),
        State([$Rew_t, Ter_t, Obs_(t+1) gets$ step environment with $Act_t$]),
        State(
          $Policy(dot.c|Obs_t) gets text("one-hot")(Act_t)$,
          comment: "Use greedy policy for action logits",
        ),
      ),
    )
  },
  caption: [ Evaluating the Bellman Update Network. ],
  placement: top,
) <alg:eval-bellman-network>

==== Policy Iteration <sec:policy-iteration>
Note that acting in this way implements policy iteration, much like the
algorithms discussed in previous chapters. As the model acts, it populates its
own context with new actions and action-logits. Since the model has been trained
on a variety of policies, it conditions its value estimates on these actions and
logits and by transitivity, on the behavior policy. When we choose actions
greedily, we improve on this behavior policy, completing the policy iteration
cycle. Note that in practice, the context of the model will contain a mixture of
older, lower-quality actions and newer, higher-quality actions, with newer
actions progressively dominating. We rely on the context-conditioning capability
of the model to approximate a policy mixing the multitude of policies
represented in the context.

=== Extension to multi-step Bellman Updates <sec:multi-step>

The present formulation trains the Bellman Update Network to perform a single
Bellman update. However, this can be generalized to multi-step updates, e.g.
using the loss:

$ Loss^delta_theta &:= -E[sum_(t=1)^(T-1)sum_(Act in Actions) log Prob_theta
(QValue_Highlight(k) (Obs_t, Act) | History^(QValue_Highlight(k- delta))_t)] $

where $delta$ is some integer between 1 angitd $k-1$ (see @eq:loss for the
definition of $History_t^(QValue_Highlight(k- delta))$). In our experiments, we
vary $delta$ between 1 and the maximum number of iterations $delta_max$. We
inversely vary $K$, the number of iterations in our evaluation (@line:iterate of
@alg:eval-tabular), so that $delta times k =delta_max$. Thus when $delta = delta_max$,
we perform $k=1$ iterations, reducing the algorithm to the "naive" method
described in @sec:naive.

== Related Work <sec:related-work-bellman-networks>
An earlier work that anticipates many of the ideas used by Bellman Update
Networks is Value Iteration Networks #cite(<tamar2016value>). Like a Bellman
Update Network, a Value Iteration Network uses a neural network to approximate a
single step of value propagation and performs multiple steps of recurrent
forward passes to produce an inference, with each new value estimate conditioned
on a previous one. However, Value Iteration Networks do not target in-context
learning, and instead the paper focuses on their ability to plan implicitly.
Additionally, Value Iteration Networks rely on Convolutional Neural Networks
which assume a representation of the environment in which the network can
simultaneously observe the current state and those adjacent to it. As a result,
the paper focuses exclusively on top-down 2D and graph navigation domains.

A more recent work that incorporates many related ideas is Procedure Cloning #cite(<yang2022chain>).
In this work, the authors augment a behavior cloning dataset with information
relating to the procedure used to choose an action. For example, in a maze
environment, instead of cloning actions only, they also clone steps in a
breadth-first-search algorithm used to choose those actions. Bellman Update
Networks may be viewed as a specialization of this approach to the policy
evaluation algorithm.

A variety of works, to include #cite(<schrittwieser2020mastering>, form: "prose"), #cite(<okada2022dreamingv2>, form: "prose"), #cite(<zhumastering>, form: "prose") and #cite(<wen2023dream>, form: "prose") consider
methods of planning in a latent space. We highlight two recent works in
particular. Thinker #cite(<chung2023thinker>) performs Monte-Carlo Tree Search
entirely in latent space, with states augmented by anticipated rollout return
and visit count. Another interesting work is #cite(<ritter2020rapid>, form: "prose") which
proposes "Episodic Planning Networks." This architecture augments the agent with
an episodic memory that gets updated using a self-attention operation that
iterates multiple times per step. The authors observe that the self-attention
operation learns a kind of "value map" of states in the environment.

== Experiments <sec:experiments-bellman-network>
Our experiments explore two settings: a tabular grid-world setting in which
ground-truth values can be computed using classical policy evaluation and a
continuous state, partially-observed domain implemented using #link("https://miniworld.farama.org/", "Miniworld") #cite(<MinigridMiniworld23>).
In the first setting, we investigate two training regimes. The first regresses
directly onto the ground-truth values, while the second incorporates the
bootstrapped training regime described in @sec:train-bellman-network. This
allows us to disentangle the effects of the iterative value estimation method at
the heart of the Bellman Update Network algorithm from the specific procedure
used to train the network.

=== Training with ground-truth values <sec:train-tabular>

#algorithm-figure(
  {
    import "algorithmic.typ": *
    algorithm(
      ..Function(
        $QValue(History)$,
        comment: "Estimate values for all states in input sequence.",
        State(
          $FormatQ(QValue_0) gets bold(0)$,
          comment: "Initialize Q-estimates to zero.",
        ),
        ..For(
          $k=0,...,K$,
          label: <line:iterate>,
          ..For(
            $Obs_t in History$,
            State(
              [$#Value_k (Obs_t) gets sum_Act Policy(Act | dot.c) #QValue_k (Obs_t, Act)$],
              comment: [Compute #FormatV("values") from #FormatQ("Q-values")],
            ),
          ),
          State(
            [$History^(#Value_k) gets (Act_t, Policy(dot.c|Obs_t), Rew_t, Ter_t, Obs_(t+1), #Value_k (Obs_(t+1)) )_(t=0)^Recency$ ],
            comment: [pair transitions with #FormatV("values")],
            label: <line:pair>,
          ),
          State(
            [$FormatQ(QValue_(k+1)) gets #QValue_theta (History^(#Value_k))$],
            comment: [Use #FormatBUN("Bellman Update Network") to estimate #FormatQ("values")],
          ),
        ),
        Return($FormatQ(QValue_(K+1))$),
      ),
    )
  },
  caption: [ Tabular evaluation of the Bellman Update Network. ],
  placement: top,
) <alg:eval-tabular>
When regressing onto ground-truth values, we simply minimize
@eq:loss-bellman-update-network, regressing onto ground-truth values for $QValue^n
(Obs^n_t, Act)$. Since we are able to optimize the value estimates for all
actions, we dispense with masking (recall the discussion in @sec:implementation)
and positional encodings. This allows us to train estimates for all states and
actions in the sequence simultaneously. To evaluate this network, we adapt the
iterative procedure from @alg:eval-bellman-network: we iteratively apply the
network first to an initial $QValue$ estimate, then auto-regressively to its own
output (after computing value estimates from the Q estimates and the policy
logits $Policy(dot.c|Obs_t)$ per @eq:value). For details see @alg:eval-tabular.

==== Do value functions overfit?
#figure(
  image("figures/bellman-update-networks/no-walls-rmse.png"),
  placement: top,
  caption: [Comparison of root mean-square error for training vs. testing.],
) <fig:root-mean-sq-error>
The first point that we wish to demonstrate in this setting is that values
conditioned on many policies are prone to overfitting. We therefore set $delta = delta_max$
(@sec:multi-step) and train the network on 80,000 randomly sampled policies in a $5 times 5$ grid-world,
with a single goal of achievement. In this idealized setting, we provide the
network with the full cross-product of states and actions, so that perfect
estimation is possible. We evaluate the network in an identical setting but with
20,000 heldout policies. As the upper-left graph of @fig:root-mean-sq-error
illustrates, we observe a significant gap between training accuracy and test
accuracy, as measured by root mean-square error. In addition, we observe that
test error mostly plateaus after update 100,000, even as training error
continues to decrease, indicating that all learning after this point entails
memorization. In the right two graphs of @fig:root-mean-sq-error, we randomly
omit $1/4$ and $1/2$ of the state-action pairs from the input. As the figures
demonstrate, the gap between training and testing widens slightly and the extent
of memorization increases.

==== Does value prediction with a Bellman Update Network mitigate overfitting?
In the lower half of @fig:root-mean-sq-error, we compare values estimated by the
Bellman Update Network. Note that the "test" lines in @fig:root-mean-sq-error
describe error for the _full_ value estimate produced by $delta_max$ steps of
iteration (following the procedure described in @sec:train-tabular), not the
error for a single Bellman update. As the figure demonstrates, test error
continues to diminish along with the training error, long after the test error
for the $delta_max$ model has plateaued. While we observe a slight diminution in
performance as the number of omitted state-action pairs increases, the gap
between train and test remains constant.

==== Do values predicted by a Bellman Update Network inform good policies?
#figure(
  image("figures/bellman-update-networks/no-walls-regret.png"),
  placement: bottom,
  caption: [Improved policy regret in the $5 times 5$ grid-world, for different values of $delta$ and
    different numbers of omitted state-action pairs.],
) <fig:improved-policy-regret>

The utility of a value function is not entirely captured by its accuracy, since
an inaccurate value function can still induce a good policy. We therefore
introduce the following procedure for evaluating value estimates in a tabular
setting:

1. We perform a single step of policy improvement, choosing actions greedily by
  value estimate.
2. We use tabular policy evaluation to evaluate the resulting policy.
3. We compare the resulting value with the value of the optimal policy.

We refer to this metric as "improved policy regret." Note that this bears some
resemblance to the procedure described in @sec:downstream and
@alg:eval-bellman-network. However, the procedure does not require the model to
auto-regressively consume the actions (and resulting transitions) produced by
the new greedy policy and consequently there is no in-context learning.

As @fig:improved-policy-regret demonstrates, all models achieve good performance
in this relatively simple setting. However, lower values of $delta$ consistently
outperform $delta_max$, indicating that the disparity in accuracy from
@fig:root-mean-sq-error does translate into performance. In general, $delta=1$ matches
or slightly outperforms the higher values of $delta$.

==== Can Bellman Update Networks generalize to novel tasks? <sec:novel-tasks>

#align(
  center,
  figure(
    grid(columns: (auto, auto), column-gutter: 10pt, [#figure(
        align(
          center,
          image("figures/bellman-update-networks/walls-rmse.png", height: 90pt),
        ),
        caption: [Root mean-square error on $5 times 5$ grid-world with walls.],
      )<fig:walls-rmse>], [#figure(
        align(
          center,
          image("figures/bellman-update-networks/walls-regret.png", height: 90pt),
        ),
        caption: [Improved policy regret on $5 times 5$ grid-world with walls.],
      )<fig:walls-regret>]),
    caption: none, // [$5 times 5$ grid-world with walls. See @sec:novel-tasks for detailed description.],
    supplement: none,
    kind: "none",
    placement: top,
    outlined: false,
  ),
)

Next we consider the effect of distribution shift in the environment dynamics
from train to test. To this end, we introduce random walls into the grid-world
with 25% probability per edge. The model does not observe walls and must infer
their presence based on the transition patterns in the inputs --- if the agent
fails to move into an adjacent grid, this indicates the presence of a wall.

During testing, we evaluate the model on a randomly generated maze. This ensures
that all grids are reachable, unlike the training setting in which grids may be
walled off in some cases. As @fig:walls-regret indicates, the model achieves
better generalization performance when trained with lower values of $delta$. We
also observe a similar generalization gap in @fig:walls-rmse as in
@fig:root-mean-sq-error.

=== Training without ground-truth targets

Until this point we assumed access to a set of ground-truth values computed
using tabular methods. In most realistic, non-tabular settings, we cannot make
this assumption. In this section, we turn our attention to the algorithm
proposed in @sec:train-bellman-network, which uses a combination of
curriculum-based training and bootstrapping to train the Bellman Update Network.

@alg:train-bellman-network introduces a handful of difficulties not present in
the previous section:

- The curriculum training approach introduces issues of non-stationarity.
- We can no longer assume complete coverage of state-action space, nor the
  capacity to sample this space IID, as we did in earlier experiments.
- The use of causal masking, as discussed in @sec:implementation, further limits
  the information on which the model may condition its predictions.

In this section we test the ability of the transformer architecture to meet
these challenges.

==== Can training without ground-truth targets yield accurate predictions? <sec:accurate-predictions>

#figure(
  grid(
    columns: (auto, auto),
    column-gutter: 20pt,
    [#figure(
        align(center, image("figures/bellman-update-networks/bootstrap-rmse.png")),
        caption: [Root mean-square error of Bellman Update Network trained without ground-truth
          targets using @alg:train-bellman-network.],
      )<fig:bootstrap-rmse>],
    [#figure(
        align(center, image("figures/bellman-update-networks/sequence-obs.png")),
        caption: [Example observation from Miniworld environment described in @sec:miniworld],
      )<fig:sequence-obs>],
  ),
  caption: none,
  supplement: none,
  kind: "none",
  placement: top,
  outlined: false,
)

Our first set of experiments in this new setting reproduces those the previous
section, with a $5 times 5$ grid-world using goals of achievement. Again, in
order to give meaning to the accuracy estimates in @fig:bootstrap-rmse, we
compare against a simple baseline, analogous to $delta_max$, which directly
estimates $QValue_(k= infinity)$. This baseline uses the same procedure as the
original (@alg:train-bellman-network), except for two changes. First, we
eliminate the curriculum. Second, we eliminate $k$ from our loss, minimizing $ Loss_theta := (QValue_theta (Obs_t, Act_t | History_t) - (Rew_t + gamma
E_(Obs_(t+1), Act_(t+1)) [QTar_theta (Obs_(t+1), Act_(t+1))]))^2 $

instead of @eq:loss-bellman-update-network. Here, $QTar_theta$ is the output of $QValue_theta$ interpolated
with previous values. To mitigate instability, we found it necessary to reduce
the interpolation factor from 0.5 to 0.001. While lower values of $delta$ clearly
outperform higher values in @fig:bootstrap-rmse, we do observe some overfitting
toward the end of training for $delta=1$.

==== Can a Bellman Update Network induce in-context reinforcement learning in a non-tabular setting? <sec:miniworld>

Finally, we turn our attention to a non-tabular setting, implemented using #link("https://miniworld.farama.org/", "Miniworld") #cite(<MinigridMiniworld23>).
Miniworld is a 3D domain in which the agent receives egocentric, RGB
observations of the environment (see @fig:sequence-obs). We adapt the #link("https://miniworld.farama.org/environments/oneroom/", "OneRoom") environment
to support multi-task training. We populate the environment with two random
objects which the agent must visit in sequence. We encode the high-dimensional
RGB observations used by Miniworld with a 3-layer convolutional network before
feeding them into the GRU transition encoder (@sec:implementation). To perform
evaluations in this setting, we use the auto-regressive rollout mechanism
described in @alg:eval-bellman-network, choosing actions greedily by value
estimate and feeding new transitions back into the network.

#figure(
  image("figures/bellman-update-networks/miniworld.png"),
  placement: bottom,
  caption: [In-context reinforcement learning curves for Bellman Update Network and
    Conservative Q-Learning (CQL).],
) <fig:miniworld>

In our Miniworld experiments, we compare three settings of $delta$ for the
Bellman Update Network. Note that for $delta=delta_max$, we eliminate the
curriculum as discussed in @sec:accurate-predictions. We also compare these with
the Conservative Q-Learning (CQL) algorithm #cite(<kumar2020conservative>), a
state-of-the-art offline RL algorithm. Unlike the Bellman Update Network, CQL
integrates policy improvement into its loss function, minimizing

$ sum_(t=1)^(T-1) (QValue_theta (Obs_t, Act_t | History^(Value_k)_t) - Highlight(max_Act) QTar(Obs_t, Highlight(Act)))^2 $
<eq:loss-cql>

where $QTar$ is computed using bootstrapping. Note that the $arg max$ operation
cannot be used in the loss when $k< infinity$. Therefore this form of policy
improvement is not available to Bellman Update Networks. A well-known property
of these kinds of losses is that they tend to produce overly optimistic value
estimates, due to the sensitivity of the $max_a$ operator to noise. To mitigate
this, CQL introduces a "conservative" auxiliary loss:

$ alpha sum_(t=1)^(T-1) log sum_Act exp (QValue(Obs_t, Act)) - QValue(Obs_t, Act_t) $
<eq:reg-cql>

Thus the algorithm used to train CQL departs from @alg:train-bellman-network in
only two ways:

1. We use the $max_a$ targets from @eq:loss-cql in place of the empirical targets
  from @eq:loss-bellman-update-network.
2. We add the regularizer from @eq:reg-cql to our loss function.

In @fig:miniworld, we compare CQL with the Bellman Update Networks under several
settings of $delta$. We also compare the results of training on different
quantities of data by stopping the training of the source algorithm at different
points in time. For reference, the middle graph in @fig:miniworld, trained on
24,576 timesteps of data, terminates training just before the source algorithm
reaches optimal performance.

Both CQL and $delta=delta_max$ experience some instability for the lower data
regimes, where the disparity in distribution between the policies represented in
the training data and the optimal policy is greatest. Moreover, we observe that
CQL and $delta=delta_max$ also learn more gradually, perhaps reflecting
limitations in the ability to generalize to the mixture policy observed during
downstream evaluation. We also observe a slight advantage for $delta=1$ over $delta=2$ in
the higher-data regimes, reflecting the disparity observed in our earlier
grid-world results.

==== Qualitative analysis

We conclude by performing some qualitative analysis on the values learned by CQL
and those learned by the Bellman Update Network. In @fig:reward-trajectory and
@fig:no-reward-trajectory, we visualize the value estimates of $delta=1$ and CQL
on two trajectories from the offline data. In the diagram, the arrows represent
the path taken by the agent. The double-headed arrows are color-coded to
indicate predicted value (for the fore arrowhead) and experienced reward (for
the aft arrowhead). The rings indicate the radius around the objects into which
the agent must enter to receive reward. In @fig:reward-trajectory, the agent
receives a cumulative reward of two, one for entering the perimeter of the blue
circle and one for entering the perimeter of the red circle.

Note that the value predictions of the Bellman Update Network approach the
maximum near the point where reward is actually received and then gradually
anneal (as indicated by the yellow, orange, and red arrowheads). By contrast,
the CQL predictions all appear to be at the maximum. Next we consider
@fig:no-reward-trajectory, depicting a trajectory in which the agent receives no
reward. Here we see much lower value predictions by CQL even as the agent
approaches the rewarding blue circle. In contrast, the predictions made by the
Bellman Update Network are similar in distribution to those in
@fig:reward-trajectory. This provides one example in which CQL anticipates the
remainder of a trajectory, implying memorization. The Bellman Update Network
does not anticipate the remainder of the trajectory and its value predictions
reflect the dynamics of the environment.

#figure(
  grid(
    columns: (auto, auto),
    column-gutter: 10pt,
    [#figure(
        grid(
          columns: (auto, auto),
          image("figures/bellman-update-networks/delta1-reward.png"),
          image("figures/bellman-update-networks/cql-reward.png"),
        ),
        caption: [Predictions by the $delta=1$ variant (left) and by CQL (right) on an offline
          trajectory with cumulative return of #Highlight([2]).],
      )<fig:reward-trajectory>],
    [#figure(
        grid(
          columns: (auto, auto),
          image("figures/bellman-update-networks/delta1-no-reward.png"),
          image("figures/bellman-update-networks/cql-no-reward.png"),
        ),
        caption: [Predictions by the $delta=1$ variant (left) and by CQL (right) on an offline
          trajectory with cumulative return of #Highlight([0]).],
      )<fig:no-reward-trajectory>],
  ),
  placement: top,
  outlined: false,
  kind: "none",
  supplement: none,
)

== Conclusion

This chapter presents an algorithm for performing in-context reinforcement
learning. It imports many of the concepts from preceding chapters, especially
the integration of context-based learning into the policy iteration algorithm.
The chapter builds on the work presented in the preceding chapters by freeing
the algorithm from model-based learning and monte-carlo rollouts. We observe
that Bellman Update Networks are better equipped to handle high-dimensional
observation spaces (like Miniworld) than AD++, since observations of this kind
pose significant challenges for existing modeling approaches, especially where
partial observability is involved. Certainly observations of this size cannot be
modeled incrementally using the inline, sequence-based approach proposed by #cite(<janner_offline_2021>, form: "prose"),
since a single observation would consume an entire context window.

That said, AD++ retains some advantages over Bellman Update Networks. In
particular, this approach may struggle to propagate values over very long
timesteps (e.g. over 100), and certainly training a network for such a task
could take a very long time. It is likely that such a setting would benefit from
values of $delta$ higher than $delta=1$, and this should be thought of as a
parameter to tune.

In general, a limitation of the approach proposed in this chapter is that it
requires learning values for a very large number of policies, whereas only the
optimal is ultimately of interest. However, learning only the optimal policy is
in general not possible within the paradigm of in-context reinforcement
learning, which requires an algorithm to yield a spectrum of policies
transitioning from exploratory to exploitative behavior.

// #bibliography("main.bib", style: "american-society-of-civil-engineers")