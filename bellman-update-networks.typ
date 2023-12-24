#import "math.typ": *
#import "@preview/big-todo:0.2.0": todo
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms

#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show: show-algorithms

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
that contributed to the expected value — the "functional" part of the
observation. Conversely, error in modeling any part of the observation, whether
functional or not, could throw off the value estimate: a model trained on inputs
containing only ground-truth observations might fail to generalize to
observations corrupted by modeling error.

In this work, we attempt to address these issues by proposing a method for
estimating value directly instead of using model-based rollouts. We will begin
by reviewing the methodology of our previous work. We will then show that this
methodology can be extended to value estimation. We will offer a critique of
this extended approach as naive. Finally, we will propose an alternative to the
naive approach which addresses some of its shortcomings.

== Review of In-Context Model-Based Planning
<review-of-in-context-model-based-planning>
In the preceding chapter, we described work in which we trained a causal model
to map a temporally sequential history of states $Obs_(<= t)$, actions
$Act_(<= t)$, and rewards $Rew_(<t)$, to predictions of the next state $Obs_(t+1)$ and
next reward $Rew_(t)$. Our model optimized the following loss:

$ Loss_theta := -sum_(n=1)^N sum_(t=1)^(T-1) log Prob_theta (Act^n_t |
History^n_t) + log Prob_theta (Rew^n_t, Ter^n_t, Obs^n_(t+1) | History^n_t,
Act^n_t) $

Here, $History^n_t$ is a trajectory drawn from the $n^"th"$ task/policy pair:
$ History^n_t := (Act^n_(t-Recency),
Rew^n_(t-Recency), Ter^n_(t-Recency), Obs^n_(t-Recency+1), dots,
Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t) $ <eq:bellman-network-history>
We implemented our model as a causal transformer
#cite(<vaswani2017attention>). Because the model was conditioned on history, it
demonstrated the capability to adapt in-context to settings with novel
transition or reward functions.

== Naive Alternative Method
<extension-to-naive-approach>
An almost identical technique might be used to predict value functions in place
of next-states and next-rewards. In principle, a history of states, actions, and
rewards should be sufficient to infer the dynamics and reward of the environment
as well as the current policy, all that is necessary to estimate value. Our
model would then minimize the following loss:

$ Loss_theta &:= -sum_(n=1)^N sum_(t=1)^(T-1)sum_(Act in Actions) log Prob_theta (Highlight(QValue^n (Obs^n_t, Act)) |
History^n_t)
$ <eq:loss1>

where $Obs^n_t$ is the last observation in $History^n_t$ and $QValue^n (Obs^n_t, Act)$ is
the value of observation $Obs^n_t$ and action $Act$ under the $n^"th"$ task/policy.
We note that in general, ground-truth targets are not available for such a loss.
However, they may be approximated using bootstrapping methods as in #cite(<mnih2015human>, form: "prose") or #cite(<kumar2020conservative>, form: "prose").

One question that we seek to understand is the extent to which this approach
scales with the number of tasks and policies, and the extent to which it
generalizes to novel tasks and policies. We observe that the mapping from a
history of states, actions, and rewards to values is non-trivial, requiring the
model to infer the policy and the dynamics of the environment, and to implicity
forecast these estimates for multiple time steps. As a result, it is reasonable
to anticipate some degree of memorization.

== Proposed Method <sec:bellman-network-method>
In the case of context-conditional models such as transformers, one factor that
has a significant impact on memorization is the extent to which the model
attends to information in its context—"in-context" learning—as opposed to the
information distilled in its weights during training—"in-weights" learning #cite(<chan2022data>).
The former will be sensitive to new information introduced during evaluation
while the latter will not. As #cite(<chan2022data>, form: "prose") suggests, the
relevance or predictive power of information in a model's context strongly
influences the balance of in-weights vs. in-context learning.

With this in mind, we note that the history of values associated with states in
the models context would be highly predictive of the value that the model is
learning to infer. A model provided this information in its context should
attend more strongly to its context and memorize less. This would entail the
following redefinition of $History^n_t$, the history on which we condition the
model's predictions:

$ History^n_t := (Act^n_(t-Recency),
Rew^n_(t-Recency), Ter^n_(t-Recency), Obs^n_(t-Recency+1), Highlight(Value^n (Obs^n_(t-Recency+1))) dots,
Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t, Highlight(Value^n (Obs^n_t))) $ <eq:bellman-network-history>

The question remains how such input values might be obtained. After all, if
these value estimates were already in hand, why would it be necessary to infer $QValue^n (Obs^n_t, Act)$ to
begin with? To address these questions, we propose an alternative approach to
predicting values _directly_ and instead take inspiration from the classic
Policy Evaluation algorithm, which iteratively improves a value estimate using
the Bellman update equation:

$ QValue_Highlight(k)^n (Obs_t^n, a_t^n) := r_t^n + gamma E[QValue_Highlight(k-1)^n (Obs_t^n, a_t^n)] $

For any $QValue_0^n$ and sufficiently large $k$, this equation is guaranteed to
converge to the true value of $QValue^n$. We incorporate a similar approach in
our method, proposing the following loss function:

$ Loss_theta &:= -sum_(n=1)^N sum_(t=1)^(T-1)sum_(Act in Actions) log Prob_theta (QValue_Highlight(k)^n (Obs^n_t, Act) |
History^n_(t, Highlight(k-1)))
\
#text("where")
History^n_(t,Highlight(k-1)) &:= (Act^n_(t-Recency),
Rew^n_(t-Recency), Ter^n_(t-Recency), Obs^n_(t-Recency+1), QValue_Highlight(k-1)^n (Obs^n_(t-Recency+1), dot.c) dots,
Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t, QValue_Highlight(k-1)^n (Obs^n_t, dot.c))
\
$ <eq:loss>

We call a model $P_theta$ that minimizes this loss a Bellman Update Network.
Initially, we may set $QValue_0^n$ to any value. By feeding this initial input
into the network and then autoregressively feeding the outputs back in, we may
obtain an estimate of both the context values and the target value in @eq:loss.

== Training procedure
Here we describe our procedure for training a Bellman Update Network. We assume
assume that we are given a dataset of states, actions, rewards, terminations,
and policy logits. We adopt a curriculum-based approach in which we initially
train
$QValue_1$ bootstrapped from $QValue_0$, which we set to *$0$*. We proceed
iteratively through higher order values until $QValue_K approx QValue_(K-1)$. At
each step in the curriculum, we continue to train $QValue_k$ for all values of $k in {1, ..., K}$ (see
@line:for-k of @alg:train-bellman-network). Otherwise, error will continue to
compound at each stage of the curriculum. Furthermore, backward transfer from
later stages may benefit earlier stages.

As the network produces estimates, we use them both to train the network (see
@line:optimize) but also to produce bootstrap targets for higher values of $k$ (see
@line:bootstrap).

Note that we may use the policy logits to compute the expectation explicity for
@line:bootstrap of @alg:train-bellman-network (@line:vest), rather than relying
on sample estimates. We also found that conditioning on these value estimates
(@line:pair) (rather than the full array of Q-values) improved the speed and
stability of learning.

#algorithm-figure(
  {
    import "algorithmic.typ": *
    algorithm(
      Input($Recency, Buffer$, comment: [Context length, RL data ]),
      State($QValue_0 gets bold(0)$, comment: "Initialize Q-estimates to zero."),
      State($K gets 0$),
      ..Repeat(
        ..Repeat(
          ..For(
            $k sim {0, ..., K}$,
            label: <line:for-k>,
            State($History sim Buffer$, comment: "sample sequence from data."),
            State(
              [$VEst_k gets sum_Act Policy(Act | dot.c) QValue_k (Obs_t, Act)$],
              comment: [Compute $E_(Act sim Policy(dot.c | Obs_t)) [QValue_k (Obs_t, Act)]$],
              label: <line:vest>,
            ),
            State(
              [$History_VEst_k gets$ pair transitions with $VEst_k$ estimates],
              label: <line:pair>,
            ),
            ..For(
              $t=0, ..., Recency$,
              State(
                $QTar_(k+1) (Obs_t, Act_t) gets Rew_t + (1-Ter_t) gamma VEst_k (Obs'_t)$,
                comment: "Bootstrap target for observed actions.",
                label: <line:bootstrap>,
              ),
            ),
            State(
              $QValue_(k+1) gets Prob_theta (History_VEst_k)$,
              comment: [Use Bellman Network to estimate values],
              label: <line:forward>,
            ),
            State(
              [minimize $sum_t (QValue_(k+1)(Obs_t, Act_t) - QTar_(k+1) (Obs_t, Act_t))^2$],
              comment: "Optimize predictions",
              label: <line:optimize>,
            ),
          ),
          $QValue_(k+1) approx QTar_(k+1)$,
        ),
        State($K gets K + 1$),
        $QValue_(k + 1) approx QValue_k$,
      ),
    )
  },
  caption: [ Training the Bellman Network. ],
) <alg:train-bellman-network>

We implement the Bellman Network as a causal transformer, using the GPT2 #cite(<radford2019language>)
implementation from #link("https://huggingface.co/", [Huggingface]). Why is
causal masking is necessary, given that the target does not appear in the input
to the model? To answer this question, we must draw attention to a disparity
between the outputs from the model on @line:forward of
@alg:train-bellman-network and the targets used to train the model on
@line:optimize. For each input state $Obs_t$, we require the model to infer a
vector of values, *$QValue(Obs_t, dot.c)$*, one for each action in the action
space. However, we are only able to train the model on the single action
observed in the dataset for that transition. If the model is able to observe
both the input state $Obs_t$ and the action $Act_t$ on which we condition the
target value, the model will neglect all predictions besides $QValue(Obs_t, Act_t)$.
That is, it will learn good predictions of $QValue(Obs_t, Act)$ for $Act = Act_t$,
the action that appears in the dataset, but not for the other actions in the
action space. To prevent this degenerate outcome, we use masking to prevent the
model from observing $Act_t$ when conditioning on $Obs_t$.

Another detail worth mentioning is that the bootstrap step on @line:bootstrap of
@alg:train-bellman-network leads to instability when generating targets for
lower targets of $k$ which the model has previously trained on. To mitigate
this, we interpolate $QEst_(k+1)$ with its previous value, using an
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
        comment: [Context length, number of iterations, length of evaluation ],
      ),
      State($QValue_0 gets bold(0)$, comment: "Initialize Q-estimates to zero."),
      State(
        [$History sim$ random behavior],
        comment: "Fill transformer context with random behavior.",
      ),
      State([$Obs_Recency gets$ reset environment]),
      ..For(
        $t=Recency, ..., Recency + T$,
        ..For(
          $k=0,...,K$,
          State(
            [$VEst_k gets sum_Act Policy(Act | dot.c) QValue_k (Obs_t, Act)$],
            comment: [Compute $E_(Act sim Policy(dot.c | Obs_t)) [QValue_k (Obs_t, Act)]$],
          ),
          State([$History_VEst_k gets$ pair transitions with $VEst_k$ estimates]),
          State([$QValue_k (Obs_t, dot.c) gets Prob_theta (History_VEst_k)$]),
        ),
        State(
          $Act_t gets arg max QValue_K (Obs_t, dot.c)$,
          comment: "Choose the action with the highest value.",
        ),
        State([$Rew_t, Ter_t, Obs_(t+1) gets$ step environment with $Act_t$]),
        State($Policy(dot.c|Obs_t) gets text("one-hot")(Act_t)$),
        State(
          $History gets History union (Act_t, Policy(dot.c|Obs_t), Rew_t, Ter_t, Obs_(t+1))$,
          comment: "Append new transition to context.",
        ),
      ),
    )
  },
  caption: [ Evaluting the Bellman Network. ],
) <alg:eval-bellman-network>

Note that acting in this way implements policy iteration, much like the
algorithms discussed in previous chapters. As the model acts, it populates its
own context with new actions and action-logits. Since the model has been trained
on a variety of policies, it conditions its value estimates on these actions and
logits and consequently on the behavior policy. When we choose actions greedily,
we improve on this behavior policy, completing the policy iteration cycle. Note
that in practice, the context of the model will contain a mixture mixture of
older, lower-quality actions and newer, higher-quality actions, with newer
actions progressively dominating. Thus we rely on the capability of the model to
generalize to this mixture policy and to approximate an average (ideally with
preference for the recent actions).

#todo("talk about rotation")

#bibliography("main.bib", style: "american-society-of-civil-engineers")
