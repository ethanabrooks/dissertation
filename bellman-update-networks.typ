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
// == Architecture
// We implement $Prob_theta$ as a transformer. In principle, the inputs to the
// transformer need not be chronological, and indeed by breaking temporal
// correlation through random sampling, it is tempting to imagine that this might
// enable us to remove masking.

// Concretely, we propose conditioning each prediction on a sequence of
// transitions sampled from trajectory $n$, which may contain multiple episodes but
// only one task and one policy. We pair each transition with an estimate of its
// Q-value. Let \$\\Q^{n}\_{\\Highlight{k+1}}\[\\Obs^n\]\[\\Act^n\] :\=
// \\RewFunc^{n}\(\\Obs^n, \\Act^n) + \\gamma \\Ex\_{\\NextObs^n, \\NextAct^n}\[
// \\Q\_{\\Highlight{k}}^{n}\[\\NextObs^n\]\[\\NextAct^n\] \]\$, where $gamma$ is
// some discount factor, \$\\RewFunc^{n}\$ is the reward function for trajectory $n$,
// and \$\\NextObs^n\$ and \$\\NextAct^n\$ are the next observation and next action
// in that trajectory. We defer discussion of \$\\Q\_{\\Highlight{0}}^{n}\$ to a
// later section. We repurpose the subscript to group variables within a
// transition, rather than indexing time. Given this notation, we propose the
// following loss:

// \$\$\\begin{aligned} \\Loss\(\\theta) & :\= -\\sum\_{n\=1}^N \\log
// \\Prob\_\\theta \[\\Qi\_{\\Highlight{k+1}}\_1^n, \\dots,
// \\Qi\_{\\Highlight{k+1}}\_n\] \[\\History^{n}\_{\\Highlight{k}} \]
// \\locallabel{eq:loss} \\intertext{where} \\begin{split}
// \\History^n\_{\\Highlight{k}} :\= &\\left\( \\Obs^n\_1, \\Act^n\_1, \\Rew^n\_1,
// \\NextObs^n\_1, \\Qi\_{\\Highlight{k}}\_1, \\right.\\\\ &\\left.\\;\\;
// \\Obs^n\_2, \\Act^n\_2, \\Rew^n\_2, \\NextObs^n\_2, \\Qi\_{\\Highlight{k}}\_2,
// \\right.\\\\ &\\left.\\;\\; \\dots\\right.\\\\ &\\left.\\;\\;
// \\Obs^n\_{\\Length}, \\Act^n\_{\\Length}, \\Rew^n\_{\\Length},
// \\NextObs^n\_{\\Length}, \\Qi\_{\\Highlight{k}}\_{\\Length}, \\right.\\\\
// &\\left.\\;\\; \\Obs^n\_{\\Length+1}, \\Act^n\_{\\Length+1},
// \\Qi\_{\\Highlight{k}}\_{\\Length+1} \\right) \\end{split}
// \\locallabel{eq:history} \\end{aligned}\$\$ \$\\History^n\_{\\Highlight{k}}\$
// contains $n$ IID sample transitions: \$\\Obs^n\_i, \\Act^n\_i, \\Rew^n\_i,
// \\NextObs^n\_i, \\NextAct^n\_i\$ is the \$i^\\Th\$ transition from the
// \$n^\\Th\$ trajectory; Note that \$\\Obs^n\_{i+1}\$ does not necessarily follow
// \$\\Obs^n\_{i}\$ in time. \$\\Length\$ is some hyperparameter determining the
// number of transitions in the prompt. Since reward and next observation are
// unknown in the downstream setting, we reserve the \$\\Length^\\Th\$ index for a
// partial transition containing only \$\\Obs^n\_\\Length\$ and
// \$\\Act^n\_\\Length\$ \(more explanation in the paragraph).

// #bibliography("main.bib", style: "american-society-of-civil-engineers")

// \$\$\\begin{aligned} \\Loss\(\\theta) & :\=
// -
// \\sum\_{n\=1}^N \\sum\_{t\=1}^{T-1} \\log \\Prob\_\\theta \[\\Qi\_{}\_t^n\] \[
// \\History^n\_t \] \\locallabel{eq:loss2} \\intertext{where} \\History^n\_t & :\=
// \\left\( \\Obs^n\_{1}, \\Act^n\_{1}, \\Rew^n\_{1}, \\Highlight{\\Qi\_{}\_{1}^n},
// \\Obs^n\_{2}, \\Act^n\_{2}, \\Rew^n\_{2}, \\Highlight{\\Qi\_{}\_{2}^n}, \\dots,
// \\Obs^n\_{t}, \\Act^n\_{t} \\right) \\end{aligned}\$\$ During training, we may
// approximate the \$\\Qi\_{}\_{\<t}^n\$ values in the prompt using our
// feed-forward value estimator \$\\Q\*\$. However, during downstream evaluation,
// the identity of the policy and task are unknown and \$\\Q\*\$ will be no use.
// Another option is monte-carlo estimation of value, which would require us to
// draw our prompt transitions from past completed episodes \(as opposed to the
// current trajectory).

// This approach requires the model to infer the value of the current observation
// and action from some batch of past transitions paired with estimates of their
// value. Implicitly, the model has to infer the full manifold of state-action
// values given samples in the prompt. In order to break spurious correlations
// between these samples, we draw them independently and identically \(IID) from
// the history of behavior. Since this manifold is quasi-continuous, this kind of
// inference is perhaps possible given sufficiently many samples.

// One concern is that monte carlo estimation may not be sufficiently accurate to
// provide useful value estimates in the downstream setting. A model trained on the
// reliable value estimates provided by \$\\Q\*\$ may fail to generalize to
// high-variance estimates provided by monte carlo. An additional concern is that
// our model might learn to approximate values without understanding the logic that
// produced them. For example, the model might learn a $k$-nearest-neighbors
// strategy of inference which produces an estimate for a given state-action pair
// by querying the prompt for the most similar state-action pair and returning its
// corresponding value. Such a strategy is unlikely to produce value estimates with
// enough precision to guide a policy that chooses actions greedily with respect to
// them.

// = Proposed Method
// <proposed-method>
// Our hypothesis is that the value estimation method which generalizes best to a
// novel downstream setting will incorporate the underlying logic of value
// propagation. Nothing in our naive approach enforces or encourages this. Our
// proposed method addresses this by learning to value iterate, rather than
// learning to estimate values directly.

// ==== Architecture
// <architecture>
// We implement \$\\Prob\_\\theta\$ as a transformer. Since our inputs are not
// indexed by time, we may omit causal masking. Also, since the input transition
// tuples are order-invariant, we may take advantage of the transformer
// architecture #cite(<lee2019set>), which reduces the complexity of self-attention
// from quadratic to linear in the context size.

// #figure(
//   [#block[
//       #block[
//         Sample trajectory \$\\Buffer\_n\$ from the dataset. \$k \\sim
//         \\Set{0}{\\dots}{K-1}\$ \$\\Obs^n\_i, \\Act^n\_i, \\Rew^n\_i, \\NextObs^n\_i
//         \\sim \\Buffer\_n\$ \$q^{\(n)}\_{k, i} \\gets \\Q\*\_k\[\\Obs^n\_i,
//         \\Act^n\_i\]\[n\]\$ \$q^{\(n)}\_{k+1, i} \\gets \\Q\*\_{k+1}\[\\Obs^n\_i,
//         \\Act^n\_i\]\[n\]\$ \$\\History^n \\gets \\Obs^n\_1, \\Act^n\_1, \\Rew^n\_1,
//         \\NextObs^n\_1, q^{\(n)}\_{k, 1}, \\dots, \\Obs^n\_\\Length, \\Act^n\_\\Length,
//         \\Rew^n\_\\Length, \\NextObs^n\_\\Length, q^{\(n)}\_{k, \\Length}
//         \\Obs^n\_{\\Length+1}, \\Act^n\_{\\Length+1}, q^{\(n)}\_{k, \\Length+1}\$
//         Maximize \$\\log \\Prob\_\\theta \[q^{\(n)}\_{k+1, 1}, \\dots, q^{\(n)}\_{k+1,
//         \\Length}\]\[\\History^n\]\$

//       ]
//     ]],
//   caption: [
//     Training the model
//   ],
// )

// ==== Training Regime
// <training-regime>
// We assume access to a collection of trajectories, indexed by task and policy.
// During training, we sample a batch of trajectories and from each trajectory,
// sample a batch of transitions. For each transition, we produce estimates
// \$\\Q\*\_{k}\[\\cdot\]\[\\cdot\]\[n\]\$ and
// \$\\Q\*\_{k+1}\[\\cdot\]\[\\cdot\]\[n\]\$ of \$\\Q^n\_{k}\$ and \$\\Q^n\_{k+1}\$
// respectively, using some traditional neural method \(more discussion in the next
// paragraph). We embed each transition separately, along with the \$\\Q\*\_{k}\$
// estimate, using a standard transformer \(no masking), then feed these embeddings
// into the transformer, with each embedding occupying one index of the input. We
// project each output of the transformer to a scalar, which we regress onto
// \$\\Q\*\_{k+1}\$ by minimizing \$\\Loss\(\\theta)\$. See Algorithm for details.

// ==== Computing \$\\Q\*\$ during Training
// <computing-q-during-training>
// \$\\Q\*\$ is to be fully trained in advance from our dataset of trajectories. We
// minimize the following standard bootstrapped mean-square loss:

// \$\$\\begin{aligned} \\Loss\_{Q}\(\\phi) & :\=
// \\Ex\_{\\Obs^n,\\Act^n,\\Rew^n,\\NextObs^n,\\NextAct^n} \[ \\Parens\( \\Rew +
// \\gamma \\Q\*\_{k}\[\\NextObs^n\]\[\\NextAct^n\]\[n; \\phi\] -
// \\Q\*\_{k+1}\[\\Obs^n\]\[\\Act^n\]\[n; \\phi\] )^2 \] \\end{aligned}\$\$
// \$\\Q\*\$ may be implemented as a multilayer perceptron. Separate networks may
// be trained for each value iteration
// $k$. Alternately, weights may be shared between networks, with indices
// corresponding to $k$ embedded and concatenated with the input.

// #figure(
//   [#block[
//       #block[
//         \$\\Obs\_i, \\Act\_i, \\Rew\_i, \\NextObs\_i \\sim \\Buffer\$ \$q\_{0, i} \\gets
//         \\QInput\_0\_i\$ \$q\_{0, \\Length+1} \\gets \\QFunc\_0\\left\(\\Obs,
//         \\Act\\right)\$ \$\\History\_k \\gets \\Obs\_1, \\Act\_1, \\Rew\_1,
//         \\NextObs\_1, q\_{k, 1}, \\dots, \\Obs\_\\Length, \\Act\_\\Length,
//         \\Rew\_\\Length, \\NextObs\_\\Length, q\_{k, \\Length} \\Obs, \\Act, q\_{k,
//         \\Length+1}\$ \$q\'\_{k, 1}, \\dots, q\'\_{k, \\Length}, q\_{k, \\Length+1}
//         \\sim \\Prob\_\\theta \[\\cdot\]\[ \\History\_k \]\$ $q prime.double_(k comma i) arrow.l k$-step
//         monte carlo estimate, starting with \$\\Rew\_i\$
//         $q_(k comma i) arrow.l lr((1 minus lambda)) q prime_(k comma i) plus lambda q prime.double_(k comma i)$
//         #strong[return] \$q\_{K, \\Length+1}\$

//       ]
//     ]],
//   caption: [
//     Computing Downstream Q-values
//   ],
// )

// ==== Downstream Evaluation
// <downstream-evaluation>
// Initially, we collect some minimum quantity of behavioral data using random
// behavior. Subsequently, we choose actions greedily with respect to value
// estimates computed using this behavioral data. See algorithm for details. This
// strategy of greedy action selection requires us to compute value estimates for
// our current observation and for all actions in our action space.

// To this end, we first form the current observation and chosen action into a
// partial transition tuple, and combine it with a set of full transitions sampled
// randomly from our behavioral data. We then use the same procedure as during
// training to embed these transitions along with some values for \$\\Q\_0^{n}\$
// and input them to our transformer. The transformer will output estimates of
// \$\\Q\_{1}^{n}\$, which we substitute for the \$\\Q\_0^{n}\$ values in our
// original inputs. Repeating this procedure $k$ times, each time with the updated
// inputs, yields \$k^\\Th\$-order value estimates for all of the transitions in
// our input, including the observation and action of interest. These value
// estimates may also be made more robust by interpolating the inferred values with
// ground-truth $k$-step monte carlo estimates, à la TD$lr((lambda))$ #cite(<sutton2018reinforcement>).
// See algorithm for details.

// ==== Choice of \$\\Q\_{0}^{n}\$
// <choice-of-q_0n>
// The transformer must be capable of generalizing its value iteration strategy
// from the training setting to the downstream setting. Therefore \$\\Q\_{0}^{n}\$
// must be similarly distributed in both settings. Additionally, the better the
// initial value estimates, fewer value iterations will be necessary to produce
// good final estimates. These desiderata suggest three choices.

// + #strong[Random Function] This choice could make good estimates for
// \$\\Q\*\_{k}\$ difficult, especially when $k$ is small, since the regression
// target would be random or close to random.

// + #strong[Random choice of \$\\mathbf{\\Q\*\_K}\$] We choose a training trajectory
// at random and pass its index to the neural estimator from the training phase.

// + #strong[Reward] Ground truth reward values could be used for all transitions
// except the one corresponding to the current observation and action. For this
// transition, we would need to train a reward model. Doing so would only require
// adding a reward prediction term to our current loss.

// Of these, the first is the most principled, since it ensures that
// \$\\Q\_{0}^{n}\$ from the training phase is identically distributed to
// \$\\Q\_{0}^{n}\$ in the downstream evaluation. However, \$\\Q\*\_{\>0}^{n}\$
// would be a more difficult regression target. The second choice might yield good
// initial estimates, but the reverse could be true depending on the setting. The
// third perhaps achieves a compromise between the two, but the need to train an
// additional reward model is a disadvantage. Empirical methods are necessary to
// determine which option is best.

// #figure(
//   [#block[
//       #block[
//         initialize \$\\Buffer\$ \$\\Obs\_0 \\gets\$ Reset environment. \$\\Act\_t \\gets
//         \\argmax\_{a}\$ \$\\Obs\_{t+1}, \\Rew\_t, \\Ter\_t \\gets\$ Execute \$\\Act\_t\$
//         in environment. $t arrow.l t plus 1$ \$\\Buffer \\gets \\Buffer \\cup \\left\(
//         \\Obs\_0, \\Act\_0, \\Rew\_0, \\Ter\_0, \\Obs\_1, \\dots, \\Obs\_t, \\Act\_t,
//         \\Rew\_t, \\Ter\_t, \\Obs\_{t+1} \\right)\$

//       ]
//     ]],
//   caption: [
//     Interacting with the Downstream Environment
//   ],
// )

// ==== Policy Improvement
// <policy-improvement>
// Our discussion up to this point has focused entirely on value estimation and
// sidestepped the question of policy improvement. As our earlier work
// #cite(<brooks2022context>) demonstrates, simply choosing actions greedily with
// respect to reasonable value estimates is sufficient to induce policy improvement
// through in-context learning and policy iteration. In order for this to occur
// however, the value estimates must track the current policy. In our setting this
// requires one small adjustment to our algorithm: transitions sampled during
// downstream evaluation must be chosen from #emph[recent] behavior — behavior
// generated by the current policy, or one close to it. Since our method is trained
// to produce value estimates for the policy which generated the input samples,
// this should be sufficient to align our value estimates with the current policy
// and induce policy iteration.

// ==== Comparison with the Naive Method
// <comparison-with-the-naive-method>
// An immediate benefit of the proposed value iteration method is that it provides
// us a way to estimate values during the downstream evaluation without resorting
// to monte carlo estimation. This will improve the quality of value estimates in
// the prompt and bring them closer in distribution to those provided to the model
// during training.

// Another benefit is that it forces our model to learn how values propagate. This
// adds signal to our loss function which penalizes course inference strategies
// like $k$-nearest-neighbors. Returning to the earlier example of a stationary
// action and an action moving toward a goal state, as value propagates from the
// goal, it will reach a point at which value has reached the move-toward action
// but not the stationary action. At this point, a wider gap than just a discount
// factor will separate the two state-action values, forcing the model to learn to
// distinguish the two based on the value propagation dynamics of the MDP. Our
// hypothesis is that a model that has absorbed this kind of knowledge will
// demonstrate more robust generalization.

// ==== Variation: Adding Lower Order Value Estimates to the Prompt
// <variation-adding-lower-order-value-estimates-to-the-prompt>
// A variation of interest is one which includes lower order estimates in addition
// to the \$k^\\Th\$-order value estimates in the prompt. This would imply the
// following updated definition of \$\\History\_{\\Highlight{k}}^{n}\$:

// \$\$\\begin{aligned} \\begin{split} \\History\_{\\Highlight{k}}^{n} :\=
// &\\left\( \\Obs^n\_1, \\Act^n\_1, \\Rew^n\_1, \\NextObs^n\_1,
// \\Qi\_{\\Highlight{1}}\_1, \\dots, \\Qi\_{\\Highlight{k}}\_1, \\right.\\\\
// &\\left.\\;\\; \\dots\\right.\\\\ &\\left.\\;\\; \\Obs^n\_\\Length,
// \\Act^n\_\\Length, \\Rew^n\_\\Length, \\NextObs^n\_\\Length,
// \\Qi\_{\\Highlight{1}}\_\\Length, \\dots, \\Qi\_{\\Highlight{k}}\_\\Length,
// \\right.\\\\ &\\left.\\;\\; \\Obs^n\_{\\Length+1}, \\Act^n\_{\\Length+1},
// \\Qi\_{\\Highlight{1}}\_{\\Length+1}, \\dots,
// \\Qi\_{\\Highlight{k}}\_{\\Length+1} \\right) \\end{split}
// \\locallabel{eq:history2} \\end{aligned}\$\$ Lower order value estimates contain
// information about the propagation of value through the MDP. For example, if grid
// $lr((0 comma 0))$ increases its value on the first value iteration and
// $lr((0 comma 3))$ increases its value on the second, this suggests the presence
// of a goal at $lr((0 comma 1))$. Since our prompt size is limited, these
// additional value estimates will compete for space with other possible inputs,
// such as additional transitions which would improve the coverage of the MDP.

// ==== Proposed Experiments and Domains
// <proposed-experiments-and-domains>
// Our first objective is to acquire "signs of life" in a simple tabular domain in
// which ground-truth value estimates can be computed quickly and reliably. In this
// setting, we wish to establish that the model is capable of inferring values for
// unseen state/action pairs, for unseen policies, and for unseen tasks. To acquire
// enough data for generalization, our domain will have to be reasonably large,
// either some variation on the four-rooms domain or some kind of programmatically
// generated set of mazes.

// Our next objective is to demonstrate that the model learns value estimates that
// are sufficiently precise to induce policy iteration. Gridworlds will be the
// first setting that we consider. In addition to the domains mentioned above, we
// propose introducing domains which are not task-identifiable, meaning that the
// task cannot be identified from a single observation. This will test the capacity
// of the policy to explore its environment before exploiting knowledge acquired
// during training.

// For a more challenging domain, we propose Alchemy
// #cite(<wang2021alchemy>), a meta-reinforcement learning benchmark released by
// DeepMind in 2021. The goal of Alchemy is to use a set of potions to transform a
// collection of visually distinctive stones into more valuable forms, collecting
// points when the stones are dropped into a central cauldron. The value of each
// stone is tied to its perceptual features, but this relationship changes from
// episode to episode, as do the potions’ transformative effects. Together, these
// structural aspects of the task constitute a "chemistry" that is fixed across
// trials within an episode, but resampled at the start of each episode.

// #figure(
//   [],
//   caption: [
//     This diagram depicts the inputs, outputs, and components of the proposed
//     architecture. For simplicity we omit the superscripts but all values in the
//     diagram are assumed to come from the same trajectory. \$\\Obs\_i, \\Act\_i,
//     \\Rew\_i,\$ and \$\\NextObs\_i\$ represent the \$i^\\Th\$ transition sampled IID
//     from this trajectory. \$\\QInput\_k\_i\$ is the \$k^\\Th\$ order state-action
//     value estimate for \$\\Obs\_i\$ and \$\\Act\_i\$. As the diagram indicates,
//     transitions are embedded by a transformer into a single summary vector and then
//     passed into a transformer which produces
//     $lr((k plus 1))$-order value estimates.
//   ],
// )
// #bibliography("main.bib", style: "association-for-computing-machinery")
