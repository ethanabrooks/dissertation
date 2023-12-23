#import "math.typ": *
#import "style.typ": cites
#import "algorithmic.typ": algorithm-figure, show-algorithms

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

Here, $History^n_t$ is a trajectory drawn from the $n^"th"$ task/policy pair. We
implemented our model as a causal transformer
#cite(<vaswani2017attention>). Because the model was conditioned on history, it
demonstrated the capability to adapt in-context to settings with novel
transition or reward functions.

== Extension to naive approach
<extension-to-naive-approach>
An almost identical technique might be used to predict value functions in place
of next-states and next-rewards. In principle, a history of states, actions, and
rewards should be sufficient to infer the dynamics and reward of the environment
as well as the current policy, all that is necessary to estimate value. Our
model would then minimize the following loss:

$ Loss_theta := -sum_(n=1)^N sum_(t=1)^(T-1) log Prob_theta (QValue^n (Obs^n_t, dot) |
History^n_t) $ <eq:loss1>

where $Obs^n_t$ is the last observation in $History^n_t$ and $QValue^n (Obs^n_t, Act)$ is
the value of observation $Obs^n_t$ and action $Act$ under the $n^"th"$ task/policy.
We note that in general, ground-truth targets are not available for such a loss.
However, they may be approximated using either monte-carlo estimates drawn from
the source data, or using bootstrapping methods as in #cite(<mnih2015human>, form: "prose") or #cite(<kumar2020conservative>, form: "prose").

// $(Obs^n_t, Act^n_t)$ under the $n^Th$ task/policy. The "ground-truth" targets

// for training might be obtained from a feed-forward neural network conditioned on
// observation, action, and $n$. We identify this task/policy-conditional network
// function with the notation \$\\Q\*\[\\Obs\]\[\\Act\]\[n\] \\approx
// \\Qi\_{}\_{}^n\$, In the downstream setting, we rely on the ability of the
// transformer model to infer the task and policy from the transitions in the input
// prompt in order to estimate value.

// In the case of context-conditional models such as transformers, one factor that
// has a significant impact on generalization is the extent to which the model
// attends to information in its context—"in-context" learning—as opposed to the
// information distilled in its weights during training—"in-weights" learning #cite(<chan2022data>).
// The former will be sensitive to new information introduced during evaluation
// while the latter will not. In this work, we adopt the assumption that more
// relevant information in the context will encourage in-context learning over
// in-weights learning.

// With this in mind, one way to increase the relevance of the context in our value
// prediction model is to include state-action values associated with each
// transition in the prompt as a kind of hint. This would entail the following
// redefinition:

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

// Concretely, we propose conditioning each prediction on a random batch of
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
#bibliography("main.bib", style: "natbib-plainnat-author-date.csl")
// #bibliography("main.bib", style: "association-for-computing-machinery")
