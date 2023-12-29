#import "style.typ": cites

#set heading(numbering: "1.1")
= Dummy <sec:introduction>

= Conclusion

Since beginning of this work, the science of artificial intelligence has
undergone a paradigm shift. Not only have foundation models, particularly
language models, come to dominate the field, but values and expectations around
research have shifted dramatically. In some ways, the work in this dissertation
aligns with these shifts. The work anticipates a world in which RL algorithms
acquire most of their knowledge through supervised training on offline datasets,
as we discussed in the Introduction (@sec:introduction). In this paradigm,
learning does not happen in a single unbroken arc, from tabula-rasa random
weights to expert behavior, all driven by end-to-end gradient-descent-based deep
RL algorithms. Instead, learning happens in _two_ stages: an initial stage in
which the model soaks up large quantities of information using strong supervised
signals from massive offline datasets, and a second stage in which the model
adapts to its specific setting using some form of in-context learning.

However, in some ways, this thesis is out of step with the current trajectory of
AI research. In particular, it focuses on some concerns which have fallen out of
favor. We consider two in particular:

1. reinforcement learning as a method of exploration and discovery.
2. generalization instead of memorization.
We will consider each of these concerns in turn, describe the ways in which
modern AI research has pivoted away from them, and make the case for their
continued relevance.

=== Reinforcement Learning as a Method of Exploration and Discovery
Reinforcement learning remains a critical component of modern AI systems, but
it's role has fundamentally changed from what its pioneers envisioned. Early
deep RL researchers imagined agents that would explore their world with very
little learning signal, progressively acquire knowledge and skills, and slowly
but surely improve their behavior until optimal. Instead of directly optimizing
for useful skills like language or object-recognition, agents would acquire
these skills in the service of a simple, sparse reward #cite(<silver2021reward>).
One of the most appealing aspects of this program for research was its
conceptual purity: nothing would be learned that did not serve the imperative of
reward. RL would gradually shed crutches like reward shaping or even auxiliary
losses #cite(<burda2018exploration>) as research matured and discovered new
pathways between short-term behavior and long-term reward.

In Yann LeCun's 2016 NeurIPS keynote #cite(<lecun2016predictive>), he laid out
an alternative paradigm, affectionately known as "LeCake":
- Unsupervised learning is the "filling" of the cake, accounting for millions of
  bits of learning per sample.
- Supervised learning is the "icing," accounting for 10-10,000 bits per sample.
- Reinforcement learning is the "cherry on top," accounting for a few bits per
  sample.
The talk caused a stir because it minimized the role of reinforcement learning,
then the dominant form of learning for AI problems. His argument was that there
was simply not enough information in reward signal to train the large neural
networks that were then coming to prominence in fields like vision.

In retrospect, the talk was incredibly prescient. Not only did the role of
reinforcement learning in large scale systems diminish, but many of the problems
that were once thought to be its exclusive purview were handed over to other
methods. Reinforcement learning still plays a critical role in the training of
LLMs, but primarily as a fine-tuning step during Reinforcement Learning from
Human Feedback (RLHF), after the bulk of training time has been spent on
unsupervised learning --- indeed the "cherry on top."

Not only does reinforcement learning lose its claim to most of the learning in
these systems, current techniques in many cases simply ignore problems that were
the very raison d'etre of reinforcement learning --- and still achieve
incredible results. While many state-of-the-art models are not publicly
documented (GPT-4, Gemini, etc.) and those that are are often cagey about the
details of RLHF #cite(<lieber2021jurassic>), it is clear that many of these
models simply ignore the credit-assignment problem by maximizing one-step reward #cite(<touvron2023llama>).
Indeed, Direct Preference Optimization (DPO) #cite(<rafailov2023direct>), one of
the most popular current techniques for optimizing learned reward, simply
eliminates credit-assignment from the loss function.

What has happening here? Six years ago, when LeCunn gave his talk, the research
community was so convinced that it needed reinforcement learning for general AI
that LeCunn prefaced his "LeCake" slide with an apology and followed the slide
with an extensive explanation of how his program might be implemented _using_ reinforcement
learning. By 2023, the "cherry on top" has arguably been reduced to sprinkles.
One reason for this shift is that the ethos of RL favors a sparse, noisy, but _aligned_ learning
signal over the now prevalent unsupervised approach, which favors a very dense,
but imperfectly aligned signal. Indeed, even in its diminished role of RLHF,
reinforcement learning is still subject to instability #cite(<ramamurthy2022reinforcement>).

Another important shift is that in many practical settings, imitation is enough.
Optimality is central to the ethos of machine learning. Once a learning problem
has been posed, we seek an optimal solution to it. Yet in many practical
settings, consistently above-average performance is adequate. The largest
language models cannot perform optimal credit assignment --- nothing in the
training regime would encourage this. However, through imitation, they inherit
the imperfect credit assignment that the humans who wrote their datasets
naturally perform. In exchange, the density of the semi-supervised learning
signal enables them to absorb knowledge and skills at a scale that is
unthinkable for the most sophisticated pure RL algorithms.

=== Generalization and Memorization
Another significant shift in thinking is the attitude toward memorization, once
thought to be synonymous with generalization. Classical frameworks like the
bias-variance tradeoff #cite(<franklin2005elements>) imply that modeling noise
will lead to overfitting --- failure to generalize from training data to test
data --- and that regularization of some kind, or truncation of training is
necessary to prevent this. However, an extensive literature has documented the
mismatch between these predictions and empirical reality #cites(<zhang2021understanding>, <brown2021memorization>).
Some work has also presented theoretical frameworks for understanding this
mismatch #cite(<feldman2020does>).

In general, there has been a movement away from classical regularization
techniques that effectively limit model expressivity in order to discourage
memorization and encourage generalization and a general recognition that these
two tendencies may be more in cooperation than in conflict #cite(<tirumala2022memorization>).
This general trend has conveniently aligned with the rise of architectures, like
the Transformer, capable of training at massive scales, and the movement toward
ever larger datasets. One crude way to characterize the overall trend is this:
rather than training on one small set of data, and then using various techniques
to generalize to a very different data set, why not train on a dataset so large
that the test data ends up being much the same as the training data?

== The continued relevance of reinforcement learning and generalization
All three chapters of this thesis concern themselves extensively with both
reinforcement learning and generalization to unseen settings. How do we justify
this focus, given the current trajectory of AI research?

First, we contend that reinforcement learning, including credit assignment, is
still an important part of the future of AI. Large Language Models continue to
grow in size, improving their performance on a variety of benchmarks testing
world knowledge and problem solving #cite(<hendrycks2020measuring>), reasoning
#cite(<ghazal2013bigbench>), math #cite(<cobbe2021training>) and coding.
However, aside from the extent of their knowledge, these models are incapable of
outperforming top experts in any field. This stands in stark contrast to
reinforcement learning techniques which, to date, stand at or near the top of
their class in several domains to include Go #cite(<silver2016mastering>),
Starcraft #cite(<vinyals2019grandmaster>), and Dota #cite(<berner2019dota>). In
order for LLMs to achieve expertise of this kind, they will either need to
revisit the problems of credit-assignment that they have chosen to disregard, or
they will need to imitate agents that do this.

Second, we argue that useful expertise will always entail a generalization
problem -- the kind where you train on one set of data and generalize to one
that is quite different. This is because expertise is always highly specialized,
at least initially. In order for their expertise to be useful, language models
will need to generalize their acquired expertise from the specific domain in
which it was acquired to other domains which may be quite different from the
first.

// #bibliography("main.bib", style: "american-society-of-civil-engineers") 