#import "math.typ": *
#import "@preview/big-todo:0.2.0": todo
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms

#set heading(numbering: "1.1")

#set math.equation(numbering: "(1)")
#set page(numbering: "1")
#show: show-algorithms
#show heading: it => text(weight: "regular", [#counter(heading).display() #smallcaps(it.body)])

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
observation #cite(<grimm2020value>). Conversely, error in modeling any part of
the observation, whether functional or not, could throw off the value estimate:
a model trained on inputs containing only ground-truth observations might fail
to generalize to observations corrupted by modeling error.

In this work, we attempt to address these issues by proposing a method for
estimating value directly instead of using model-based rollouts. We will begin
by reviewing the methodology of our previous work. We will then show that this
methodology can be extended to value estimation. We will offer a critique of
this extended approach as naive. Finally, we will propose an alternative to the
naive approach which addresses some of its shortcomings.

== Preliminaries
=== Review of In-Context Model-Based Planning <review-of-in-context-model-based-planning>

In the preceding chapter, we described work in which we trained a causal model
to map a temporally sequential history of states $Obs_(<= t)$, actions
$Act_(<= t)$, and rewards $Rew_(<t)$, to predictions of the next state $Obs_(t+1)$ and
next reward $Rew_(t)$. Our model optimized the following loss:

$ Loss_theta := -sum_(n=1)^N sum_(t=1)^(T-1) log Prob_theta (Act^n_t |
History^n_t) + log Prob_theta (Rew^n_t, Ter^n_t, Obs^n_(t+1) | History^n_t,
Act^n_t) $

Here, $History^n_t$ is a trajectory drawn from the $n^"th"$ task/policy pair:
$ History^n_t := (Act^n_(t-1-Recency),
Rew^n_(t-1-Recency), Ter^n_(t-1-Recency), Obs^n_(t-Recency), dots,
Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t) $ <eq:bellman-network-history>
We implemented our model as a causal transformer
#cite(<vaswani2017attention>). Because the model was conditioned on history, it
demonstrated the capability to adapt in-context to settings with novel
transition or reward functions.

=== Naive Alternative Method <sec:naive>

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
However, they may be approximated using bootstrapping methods as in #cite(<mnih2015human>, form: "prose")
or #cite(<kumar2020conservative>, form: "prose").

One question that we seek to understand is the extent to which this approach
scales with the number of tasks and policies, and the extent to which it
generalizes to novel tasks and policies. We observe that the mapping from a
history of states, actions, and rewards to values is non-trivial, requiring the
model to infer the policy and the dynamics of the environment, and to implicitly
forecast these estimates for multiple time steps. As a result, it is reasonable
to anticipate some degree of memorization.

== Proposed Method <sec:bellman-network-method>

In the case of context-conditional models such as transformers, one factor that
has a significant impact on memorization is the extent to which the model
attends to information in its context—"in-context" learning—as opposed to the
information distilled in its weights during training—"in-weights" learning. The
former will be sensitive to new information introduced during evaluation while
the latter will not. As #cite(<chan2022data>, form: "prose") suggests, the
relevance or predictive power of information in a model's context strongly
influences the balance of in-weights vs. in-context learning.

With this in mind, we note that the history of values associated with states in
the models context would be highly predictive of the value of the current state.
A model provided this information in its context should attend more strongly to
its context and memorize less. This would entail the following redefinition of $History^n_t$,
the history on which we condition the model's predictions:

$ History^n_t := (Act^n_(t-1-Recency),
Rew^n_(t-1-Recency), Ter^n_(t-1-Recency), Obs^n_(t-1-Recency), Highlight(QValue^n (Obs^n_(t-Recency+1), dot.c)) dots,
Highlight(QValue^n (Obs^n_t, dot.c)),
Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t, ) $ <eq:bellman-network-history>

The question remains how such input values might be obtained. After all, if
these value estimates were already in hand, why would it be necessary to infer $QValue^n (Obs^n_t, Act)$ to
begin with? To address these questions, we propose an alternative approach to
predicting values _directly_ and instead take inspiration from the classic
Policy Evaluation algorithm, which iteratively improves a value estimate using
the Bellman update equation:

$ QValue_Highlight(k) (Obs_t, Act_t) := Rew(Obs_t, Act_t) + gamma E_(Obs' Act') [QValue_Highlight(k-1) (Obs', Act')] $

For any $QValue_0$ and for sufficiently large $k$, this equation is guaranteed
to converge to the true value of $QValue$ #cite(<sutton2018reinforcement>). We
incorporate a similar approach in our method, proposing the following loss
function:

$ Loss_theta &:= -sum_(n=1)^N sum_(t=1)^(T-1)sum_(Act in Actions) log Prob_theta (QValue_Highlight(k)^n (Obs^n_t, Act) |
History^n_(t, Highlight(k-1)))
\
#text("where")
History^n_(t,Highlight(k-1)) &:= (Act^n_(t-1-Recency),

Rew^n_(t-1-Recency), Ter^n_(t-1-Recency), Obs^n_(t-Recency), QValue_Highlight(k-1)^n (Obs^n_(t-Recency), dot.c) dots,
QValue_Highlight(k-1)^n (Obs^n_(t-1), dot.c),
Act^n_(t-1), Rew^n_(t-1), Ter^n_(t-1), Obs^n_t, )
\
$ <eq:loss>

We call a model $P_theta$ that minimizes this loss a Bellman Update Network
(BUN). Initially, we may set $QValue_0^n$ to any value. By feeding this initial
input into the network and then auto-regressively feeding the outputs back in,
we may obtain an estimate of both the target value, $QValue_(k)^n (Obs^n_t, Act)$ and
the context values $QValue_(k-1)^n (Obs^n_(t-Recency+1), dot.c),..., QValue_(k-1)^n (Obs^n_(t-1), dot.c)$.

Note that this approach entails a tradeoff. By training on $k < infinity$, we
significantly increase the space of inputs for the model in a given quantity of
data. Therefore, any training procedure will be much slower, and the demands on
its representational capacity greater. In exchange, as we demonstrate in
@sec:experiments-bellman-network, we gain more robust representations, capable
of better generalization to unseen settings.

=== Training procedure <sec:train-bellman-network>

Here we describe a practical procedure for training a Bellman Update Network. We
assume that we are given a dataset of states, actions, rewards, terminations,
and policy logits. For all values of $k$ greater than 1, we must choose between
using inaccurate bootstrap targets or adopting a curriculum, which introduces
challenges of non-startionarity. We adopt the latter approach, in order to avoid
expanding the input space of the model to all possible value functions, and not
just those corresponding to policies in the dataset.

Our curriculum initially trains
$QValue_1$ bootstrapped from $QValue_0$, which we set to *$0$*. We proceed
iteratively through higher order values until $QValue_K approx QValue_(K-1)$. At
each step in the curriculum, we continue to train $QValue_k$ for all values of $k in 1, ..., K$ (see
@line:for-k of @alg:train-bellman-network). This allows the network to continue
improving its estimates for lower values of $k$ even as it begins to train on
higher values, thereby mitigating the effects of compound error inherent to
bootstrapping approaches. Another benefit of continuing to train lower values of $k$ is
that these estimates can benefit from backward transfer as the curriculum
progresses. As the network produces estimates, we use them both to train the
network (see @line:optimize) but also to produce bootstrap targets for higher
values of $k$ (see @line:bootstrap).

Note that we use the policy logits to compute the expectation explicitly for
@line:bootstrap of @alg:train-bellman-network (@line:vest), rather than relying
on sample estimates. We also found that conditioning on these value estimates
(@line:pair) rather than the full array of Q-values improved the speed and
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
            ..For(
              $Obs_t in History$,
              State(
                [$VEst_k (Obs_t) gets sum_Act Policy(Act | dot.c) QValue_k (Obs_t, Act)$],
                comment: [Compute $E_(Act sim Policy(dot.c | Obs_t)) [QValue_k (Obs_t, Act)]$],
                label: <line:vest>,
              ),
            ),
            State(
              [$History^(VEst_k) gets$ pair transitions with $VEst_k$ estimates],
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
              [$QValue_(k+1) gets QValue_theta (History^(VEst_k))$],
              comment: [Use Bellman Network to estimate values],
              label: <line:forward>,
            ),
            State(
              [maximize $sum_t [QValue_theta (Obs_t, Act_t | History^(VEst_k)_t) - QTar_(k+1)(Obs_t, Act_t)]^2 $],
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

=== Implementation Details <sec:implementation>
We implement the Bellman Network as a causal transformer, using the GPT2 #cite(<radford2019language>)
implementation from #link("https://huggingface.co/", [www.huggingface.co]). Why
is causal masking is necessary, given that the target does not appear in the
input to the model? To answer this question, we must draw attention to a
disparity between the outputs from the model on @line:forward of
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

One consequence of masking is that predictions for values early in the sequence
are poor in comparison to predictions later in the sequence, since they benefit
from less context. Repeated bootstrapping from these poor predictions propagates
error throughout the sequence. To mitigate this, we rotate the sequence by
fractions, retaining predictions from only the last fraction. For example, if we
break the sequence into three equal fractions $(X_1, X_2, X_3)$, we apply three
rotations, yielding rotated sequences $(X_1, X_2, X_3)$, $(X_2, X_3, X_1)$, and $(X_3, X_1, X_2)$.
We pass each rotation through the model, and for each rotation, we retain only
the predictions for $X_3$, $X_1$, and $X_2$ respectively. We use this rotation
procedure to produce the Q estimates on @line:forward of
@alg:train-bellman-network.

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
          ..For(
            $Obs_t in History$,
            State(
              [$VEst_k (Obs_t) gets sum_Act Policy(Act | dot.c) QValue_k (Obs_t, Act)$],
              comment: [Compute $E_(Act sim Policy(dot.c | Obs_t)) [QValue_k (Obs_t, Act)]$],
            ),
          ),
          State(
            [$History_VEst_k (Obs_t) gets$ pair transitions with $VEst_k$ estimates],
          ),
          State(
            [$QValue_(k+1) gets QValue_theta (History_VEst_k)$],
            comment: [Use the Bellman Update Network to estimate values.],
          ),
        ),
        State(
          $Act_t gets arg max QValue_(K+1) (Obs_t, dot.c)$,
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
  caption: [ Evaluating the Bellman Update Network. ],
) <alg:eval-bellman-network>

==== Policy Iteration <sec:policy-iteration>
Note that acting in this way implements policy iteration, much like the
algorithms discussed in previous chapters. As the model acts, it populates its
own context with new actions and action-logits. Since the model has been trained
on a variety of policies, it conditions its value estimates on these actions and
logits and consequently on the behavior policy. When we choose actions greedily,
we improve on this behavior policy, completing the policy iteration cycle. Note
that in practice, the context of the model will contain a mixture of older,
lower-quality actions and newer, higher-quality actions, with newer actions
progressively dominating. Thus we rely on the context-conditioning capability of
the model to approximate a policy mixing the multitude of policies represented
in the context.

=== Extension to multi-step Bellman Updates <sec:multi-step>
Note that the formulation so far discussed can be generalized to multi-step
updates, e.g. using the loss:

$ Loss_theta &:= -sum_(n=1)^N sum_(t=1)^(T-1)sum_(Act in Actions) log Prob_theta (QValue_Highlight(k)^n (Obs^n_t, Act) |
History^n_(t, Highlight(k- delta))) $

where $delta$ is some integer between 1 and $k-1$ (recall @eq:loss for the
definition of $History^n_(t, k- delta)$). In our experiments, we vary $delta$ between
1 and some maximum number of iterations $delta_max$. We inversely vary $K$, the
number of iterations in our evaluation (@line:iterate of @alg:eval-tabular), so
that $delta times k =delta_max$. Thus when $delta = delta_max$, we perform $k=1$ iterations,
and reducing the algorithm to the "naive" method described in @sec:naive.

== Experiments <sec:experiments-bellman-network>
Our experiments explore two settings: a tabular grid-world setting in which
ground-truth values can be computed using classical policy evaluation
#cite(<sutton2018reinforcement>) and a continuous state, partially-observed
domain implemented using #link("https://miniworld.farama.org/", "Miniworld") #cite(<MinigridMiniworld23>).
In the first setting, we investigate two training regimes. The first regresses
directly onto the ground-truth values, while the second incorporates the
bootstrapped training regime described in @sec:train-bellman-network. This
allows us to disentangle the effects of the iterative value estimation method at
the heart of the Bellman Update Network algorithm from the specific procedure
used to train the network.

=== Training with ground-truth values <sec:train-tabular>
When regressing onto ground-truth values, we simply minimize @eq:loss supplying
ground-truth values (computed using tabular policy evaluation) for $QValue^n (Obs^n_t, Act)$.
Since we are able to optimize the value estimates for all actions, we dispense
with masking and (recall the discussion in @sec:implementation) and positional
encodings. This allows us to train estimates for all states and actions in the
sequence simultaneously. To evaluate this network we extract the iterative
procedure from @alg:eval-bellman-network: we iteratively apply the network first
to an initial $QValue$ estimate, then auto-regressively to its own output (after
computing $VEst$ estimates from the $QValue$ estimates and the policy logits $Policy(Act|dot.c)$).
For details see @alg:eval-tabular.

#algorithm-figure(
  {
    import "algorithmic.typ": *
    algorithm(
      ..Function(
        $QValue(History)$,
        comment: "Estimate values for all states in input sequence.",
        State($QValue_0 gets bold(0)$, comment: "Initialize Q-estimates to zero."),
        ..For(
          $k=0,...,K$,
          label: <line:iterate>,
          ..For(
            $Obs_t in History$,
            State(
              [$VEst_k (Obs_t) gets sum_Act Policy(Act | dot.c) QValue_k (Obs_t, Act)$],
              comment: [Compute $E_(Act sim Policy(dot.c | Obs_t)) [QValue_k (Obs_t, Act)]$],
            ),
          ),
          State([$History_VEst_k gets$ pair transitions with $VEst_k$ estimates]),
          State(
            [$QValue_(k+1) gets Prob_theta (History_VEst_k)$],
            comment: [Use the Bellman Update Network to estimate values.],
          ),
        ),
        Return($QValue_(K+1)$),
      ),
    )
  },
  caption: [ Tabular evaluation of the Bellman Update Network. ],
) <alg:eval-tabular>

==== Do value functions overfit?
#figure(
  image("figures/bellman-update-networks/no-walls-rmse.png"),
  placement: bottom,
  caption: [Comparison of root mean-square error for training vs. testing.],
) <fig:root-mean-sq-error>
The first point that we wish to demonstrate in this setting is that values
conditioned on many policies are prone to overfitting. To illustrate this point,
we set $delta = delta_max$ (recall @sec:multi-step) and train the network on
80,000 randomly sampled policies in a $5 times 5$ grid-world, with a single goal
of achievement. In this idealized setting, we provide the network with the full
cross-product of states and actions, so that perfect estimation is possible. We
evaluate the network in an identical setting but with 20,000 heldout policies.
As the upper-left graph of @fig:root-mean-sq-error, illustrates, we observe a
significant gap between training accuracy and test accuracy, as measured by root
mean-square error. In addition, we observe that test error mostly plateaus after
update 100,000, even as train error continues to decrease, indicating that all
learning after this point entails memorization.

In the right two graphs of @fig:root-mean-sq-error, we randomly omit $1/4$ and $1/2$ of
the state-action pairs from the input. As the figures demonstrate, the gap
between training and testing widens and the extent of memorization increases.
For omitted state-action pairs, the model must infer the marginal for the
policy. Thus the mapping from context to output becomes more complex as more
state-action pairs are omitted. The resulting drop in generalization recalls the
intuition that we offered in @sec:bellman-network-method: the less informative a
model's context, the more it will revert to memorization.

==== Does value prediction with a Bellman Update Network mitigate overfitting?
In the lower half of @fig:root-mean-sq-error, we compare values estimated by the
Bellman Update Network. As the figure demonstrates, test error continues to
diminish along with the training error, long after the test error for the $delta_max$ model
has plateaued. While we observe a slight diminution in performance as the number
of omitted state-action pairs increases, the gap between train and test remains
constant.

==== Do values predicted by a Bellman Update Network inform good policies?
#figure(
  image("figures/bellman-update-networks/no-walls-regret.png"),
  placement: bottom,
  caption: [Improved policy regret in the $5 times 5$ grid-world, for different values of $delta$ and
    different numbers of omitted state-action pairs.],
) <fig:improved-policy-regret>

The utility of a value function is not entirely captured by its accuracy: an
inaccurate value function can still induce a good policy. We therefore introduce
the following procedure for evaluating value estimates in a tabular setting:

1. We perform a single step of policy improvement, choosing actions greedily by
  value estimate.
2. We use tabular policy evaluation to evaluate the resulting policy.
3. We compare the resulting value with the value of the optimal policy.

We refer to this metric as "improved policy regret." Note that this bears some
resemblance to the procedure described in @sec:downstream and
@alg:eval-bellman-network. However, this procedure does not require the model to
auto-regressively consume the actions (and resulting transitions) produced by
the new greedy policy. There is consequently no in-context learning, as the
model does not consume transitions resulting from the newly improved policy.

As @fig:improved-policy-regret demonstrates, all models achieve good performance
in this relatively simple setting. However, lower values of $delta$ consistently
outperform $delta_max$, indicating that the disparity in accuracy from
@fig:root-mean-sq-error does translate into performance. In general, $delta=1$ matches
or slightly outperforms the higher values of $delta$.

==== Can Bellman Update Networks generalize to novel tasks? <sec:novel-tasks>

#align(
  center,
  figure(
    grid(columns: (auto, auto), [#figure(
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
walled off in some cases. Generalization in this setting is possible, since the
model has the opportunity during training to learn about wall dynamics during
training. However, the general "shape" of value in the environment is quite
different during evaluation, in which walls form passages instead of scattering
randomly.

As the results in @fig:walls-regret indicates, the model achieves better
generalization performance when trained with lower values of $delta$. We also
observe a similar generalization gap in @fig:walls-rmse as in
@fig:root-mean-sq-error.

TODO: qualitative analysis

=== Training without ground-truth targets

Until this point we assumed access to a set of ground-truth values computed
using tabular methods. In most realistic, non-tabular settings, such methods are
not tractable. In this section, we turn our attention to the algorithm proposed
in @sec:train-bellman-network, which uses a combination of curriculum-based
training and bootstrapping to train the Bellman Update Network.

@alg:train-bellman-network introduces a handful of difficulties not present in
the previous section. First, the curriculum training approach introduces issues
of non-stationarity. Second, we can no longer assume complete coverage of
state-action space, nor the capacity to sample this space IID, as we did in
earlier experiments. Third, our method encounters challenges of compounding
error, though these are not fundamentally different from those encountered by
other bootstrapping methods. Finally, the use of causal masking, as discussed in
@sec:implementation, further limits the information on which the model may
condition its predictions. In this section we investigate the empirical question
of whether the transformer architecture is equal to these challenges.

==== Can @alg:train-bellman-network yield accurate predictions? <sec:accurate-predictions>
#figure(
  image("figures/bellman-update-networks/bootstrap-rmse.png"),
  placement: top,
  caption: [TODO: this is a placeholder.],
) <fig:bootstrap-rmse>
Our first set of experiments reproduces those in @sec:train-tabular. Again, in
order to give meaning to the accuracy estimates in @fig:bootstrap-rmse, we
compare against a simple baseline, analogous to $delta_max$, which directly
estimates $QValue_(k= infinity)$. In order to train this baseline, we use an
algorithm identical to @alg:train-bellman-network, except we eliminate the
curriculum, and in place of @line:optimize, we minimize the traditional
bootstrapped loss:
$ Loss_theta := (QValue_theta (Obs_t, Act_t | History_t) - (Rew_t + gamma E_(Obs_(t+1), Act_(t+1)) [QTar_theta (Obs_(t+1), Act_(t+1))]))^2 $

where $QTar_theta$ is the output of $QValue_theta$ interpolated with previous
values. To mitigate instability, we found it necessary to reduce the
interpolation factor from 0.5 to 0.001. Nonetheless, we were unable to prevent
these predictions from eventually diverging as seen in @fig:bootstrap-rmse. Some
existing literature has documented the tendency for value prediction to overfit
in the offline setting, especially when integrating policy improvement.

TODO: add visualizations diagrams

==== Can @alg:train-bellman-network induce in-context reinforcement learning in a non-tabular setting?
#figure(
  image("figures/bellman-update-networks/oneroom.jpg", width: 85%),
  placement: top,
  caption: [A screenshot of the Miniworld environment. The agent also observes objects with
    different colors and shapes.],
) <fig:pickupobjects>
Finally, we turn our attention to a non-tabular setting, implemented using #link("https://miniworld.farama.org/", "Miniworld") #cite(<MinigridMiniworld23>).
Miniworld is a 3D domain in which the agent receives egocentric, RGB
observations of the environment (see @fig:pickupobjects). We adapt the #link("https://miniworld.farama.org/environments/oneroom/", "OneRoom") environment
to support multi-task training. We populate the environment with two random
objects which the agent must visit in sequence.

TODO: discuss difference from CQL in greater depth.

TODO: explain the evaluation setting more clearly (reference
@alg:eval-bellman-network)

In @fig:miniworld, we compare three settings of $delta$ for the Bellman Update
Network. Note that for $delta=delta_max$, we eliminate the curriculum as
discussed in @sec:accurate-predictions. We also compare these with the
Conservative Q-Learning (CQL) algorithm #cite(<kumar2020conservative>), a
state-of-the-art offline RL algorithm. We compare these four algorithms on a
variety of data quantities. For reference, the middle graph in @fig:miniworld,
trained on 24,576 timesteps of data, terminates training just before the source
algorithm reaches optimal performance.

Both CQL and $delta=delta_max$ experience some instability for the lower data
regimes, where the disparity in distribution between the policies represented in
the training data and the optimal policy is greatest. Moreover, we observe that
CQL and $delta=delta_max$ also learn more gradually, perhaps reflecting
limitations in the ability to generalize to the mixture policy observed during
downstream evaluation. We also observe a slight advantage for $delta=1$ over $delta=2$,
reflecting the disparity observed in our earlier grid-world results.

#figure(
  image("figures/bellman-update-networks/miniworld.png"),
  placement: top,
  caption: [In-context reinforcement learning curves for Bellman Update Network(BUN) and
    Conservative Q-Learning (CQL).],
) <fig:miniworld>

== Related Work
TO DO.

== Discussion
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

