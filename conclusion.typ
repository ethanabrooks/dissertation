#import "style.typ": cites

= Conclusion

Since beginning of this work, the science of artificial intelligence has
undergone a paradigm shift. Not only have foundation models, particularly
language models, come to dominate the field, but priorities and expectations
around research have shifted dramatically. In some ways, the work in this thesis
aligns with these shifts. The thesis anticipates a world in which RL algorithms
acquire most of their knowledge through supervised training on offline datasets,
as we discussed in @sec:introduction[the introduction]. In this paradigm,
learning does not happen in a single unbroken arc, from tabula-rasa random
weights to expert behavior, all driven by end-to-end gradient-descent-based deep
RL algorithms. Instead, learning happens in _two_ stages: an initial stage in
which the model soaks up large quantities of information using strong supervised
signals from offline datasets, and a second stage in which the model adapts to
its specific setting using some form of in-context learning.

However, in some ways, this thesis is out of step with the current trajectory of
AI research. In particular, it focuses on two concerns which have fallen out of
favor:

1. reinforcement learning as a method of exploration and discovery.
2. generalization instead of memorization.
We will begin by describing the ways in which attitudes towards these concerns
have changed, and make the case for their continued relevance.

=== Reinforcement Learning as a Method of Exploration and Discovery
Reinforcement learning remains a component of some state-of-the-art AI systems,
but it's role has fundamentally changed from what its pioneers envisioned. Early
deep RL researchers imagined agents that would explore their world with very
little learning signal, progressively acquire knowledge and skills, and slowly
but surely improve their behavior until optimal. Agents would acquire auxiliary
skills as necessary in the service of a simple, sparse reward #cite(<silver2021reward>),
rather than optimizing those skills directly. One of the most appealing aspects
of this program was its rigorous economy in aligning objectives: all learning
would serve the accumulation of reward. An agent might acquire language, for
example, but only to the extent necessary for communicating concepts essential
to its mission. This ethos dictated that RL would gradually shed crutches like
reward shaping #cite(<hu2020learning>) and auxiliary losses #cite(<burda2018exploration>) as
research matured and discovered new pathways between short-term behavior and
long-term reward.

In Yann LeCun's 2016 NeurIPS keynote #cite(<lecun2016predictive>), he laid out a
paradigm that came to be known as "LeCake":
- Unsupervised learning is the "filling" of the cake, accounting for millions of
  bits of learning per sample.
- Supervised learning is the "icing," accounting for 10-10,000 bits per sample.
- Reinforcement learning is the "cherry on top," accounting for a few bits per
  sample.
The talk caused a stir because it minimized the role of reinforcement learning,
then the dominant form of learning for AI problems. His argument was that there
was simply not enough information in reward signal to train the large neural
networks that were then coming to prominence in fields like computer vision.

In retrospect, the talk was incredibly prescient. Not only did the role of
reinforcement learning in large-scale systems diminish, but many of the problems
that were once thought to be its exclusive purview, such as credit assignment,
are now routinely ignored. Reinforcement learning still plays a role in the
training of some LLMs, but primarily as a fine-tuning step during Reinforcement
Learning from Human Feedback (RLHF), after the bulk of training time has been
spent on unsupervised learning --- indeed the "cherry on top."

One problem that receives the most shocking neglect from modern systems and
which was once the very raison d'etre of reinforcement learning is "credit
assignment." This is the problem of determining which actions, in a temporally
extended sequence, are responsible for a final outcome, good or bad. It is
through credit assignment, for example, that we learn that losing our queen on
the 20th step of a chess game is detrimental, even though we might not actually
lose the game for another 80 turns. Technically, language entails
credit-assignment, since a particular conclusion may only be reachable through a
long sequence of thoughts, calculations or questions.

While many state-of-the-art models are not publicly documented (GPT-4, Gemini,
etc.) and others are cagey about the details of RLHF #cite(<lieber2021jurassic>),
it is clear that models increasingly ignore the credit-assignment problem by
maximizing one-step reward #cite(<touvron2023llama>) -- and despite this,
achieve top performance. Direct Preference Optimization #cite(<rafailov2023direct>),
an increasingly popular technique for optimizing learned reward, actually brands
itself as "RL-free" and completely disregards credit-assignment.

What explains the precipitous shift in the role of RL in modern AI systems? In
short, a tradeoff: between the kind of rigorous alignment that RL prioritizes
and the kind of scale necessary to train most practical AI systems. Aligning
behavior with reward requires credit-assignment. Realistic settings require
long-term credit assignment under conditions of noise and uncertainty. Learning
signal under these conditions is inherently weak. Strong learning signal is
necessary to train large networks on large amounts of data. When GPT3 #cite(<brown2020language>) demonstrated
the indisputable power of scale, approaches that were incompatible began to fall
out of favor.

Another important shift is that in many practical settings, imitation is enough.
To reiterate, many natural language tasks do entail credit assignment,
especially when multi-step, exploratory reasoning is involved #cite(<wei_chain_2022>).
However, language models inherit a kind of imperfect credit assignment through
imitation of the humans who wrote their data (who already somehow possess the
capacity for credit assignment somehow). RL provides a framework for learning
_optimal_ credit assignment, but this framework does not empirically work in
realistic language settings. Of course, an unprincipled solution that works is
better than a principled solution that doesn't.

=== Generalization and Memorization
Another significant shift in thinking is the attitude toward memorization, once
thought to be synonymous with generalization. Classical frameworks like the
bias-variance tradeoff #cite(<franklin2005elements>) imply that modeling noise
will lead to overfitting --- failure to generalize from training data to
out-of-distribution test data --- and that regularization of some kind or
truncation of training is necessary to prevent this. However, several
publications have documented the mismatch between these predictions and
empirical reality #cites(<zhang2021understanding>, <brown2021memorization>).
Some work has also presented theoretical frameworks for understanding this
mismatch #cite(<feldman2020does>).

In general, there has been a movement away from classical regularization
techniques that effectively limit model expressivity in order to discourage
memorization and encourage generalization, and a general recognition that these
two tendencies may be more in cooperation than in conflict #cite(<tirumala2022memorization>).
This general trend has conveniently aligned with the rise of learning at large
scales, which presents more opportunities for memorization due to the size of
the architectures, and fewer costs due to the size of the training data. When
training data is large enough, there is no such thing as "out-of-distribution."
When all test data already appears in some form in the training data, the
problem of generalization becomes obsolete.

== The continued relevance of reinforcement learning and generalization
All three chapters of this thesis concern themselves extensively with
temporally-extended reinforcement learning and generalization to unseen
settings. How do we justify this focus, given the current trajectory of AI
research?

=== Language privileges imitation
For all its success in the realm of language, imitative learning faces
challenges in other domains. Language lends itself especially well to imitation
--- a model can reproduce the exact words in its dataset. In contrast, other
domains do not permit this kind of exact reproduction, requiring the transfer of
knowledge and skills from one setting to a different one. In robotics, for
example, the same behavior requires widely different policies for different
physical embodiments. The joint activation patterns that a 300-lb steel robot
must use to walk are significantly different from those of a 150-lb human. For
any robot to acquire some skill from existing embodiments, in the way that an
LLM acquires language understanding from human datasets, it must overcome a
significant problem of transfer.

Zero-shot transfer of the kind exhibited by LLMs is not likely for robots,
especially if they learn through extensive memorization. Instead, some period of
fine-tuning or in-context learning will be necessary to adapt fundamental skills
acquired from offline data to specific embodiments and settings. Unlike language
models, which acquire credit-assignment strategies whole-cloth from their source
data, robots will need to adapt these strategies and learn new ones. Some
framework like RL, capable of evaluating and improving credit assignment, will
be necessary. #cite(<andrychowicz2020learning>, form: "prose") offer one
compelling approach to this problem.

=== Credit assignment is necessary for expertise
Advances in the technologies that power LLMs have been rapid and difficult to
predict. However, a general trend is that they improve the ability of models to
imitate their sources, not their ability to outperform them by any significant
margin. As a rule, LLMs do not outperform top experts in any field. This stands
in stark contrast to agents trained using reinforcement learning which, to date,
stand at or near the top of their class in several domains including Go #cite(<silver2016mastering>),
Starcraft #cite(<vinyals2019grandmaster>), and DoTA #cite(<berner2019dota>).

Many areas of expertise entail reasoning or decision-making over multiple steps.
This is especially true of many of the loftier aspirations for AI systems,
including scientific discovery and artistic creativity. A program that neglects
credit-assignment can acquire human-level credit assignment through imitation.
To a degree, it can acquire super-human expertise through fine-tuning of the
final result. However, the expertise of its final inference or output will
fundamentally be limited by the multi-step process which led to the inference.
Improving the process with respect to the final result requires some form of
credit-assignment.

=== Generalization is necessary for expertise

Finally, we argue that useful expertise will always entail transfer and
generalization -- that is, a meaningful gap between training data and the domain
of interest. Whatever their limitations in terms of expertise, LLMs
unquestionably distinguish themselves by their breadth --- their capacity to
produce coherent, usually intelligent, responses to almost any prompt. A system
that combines the expertise of RL with the breadth of LLMs does not currently
exist. Therefore, the first of its kind will need to distill expertise from
_specialists_ and then generalize it beyond those specializations. For example,
a model may acquire knowledge of advanced mathematics by training on textbooks,
or even by distilling the behaviors of RL agents specialized to certain
mathematical problems, like AlphaTensor #cite(<fawzi2022discovering>). However,
in order to advance the frontiers of science, such a model would need to
transfer this specialized intelligence to other domains, domains which are not
captured by any dataset since they potentially lie beyond the limits of current
human understanding.

=== Foundation models for reinforcement learning
A consequence of these reflections is that foundation models for RL will not
look the same as foundation models for language. "LeCake" provides a sketch that
may still apply --- surely, RL agents can benefit from data beyond their
immediate experience. However, the exact program will need to be different in
significant ways. This thesis has offered some tentative ideas about RL
foundation models. In the first chapter, we observe that any sequence-based
foundation model can, in principle, serve as a kind of RL foundation model. In
the second, we demonstrate that such a model gains capacity when specialized to
RL data. Through the incorporation of Algorithm Distillation, we highlight the
fact that such a model can not only distill the dynamics and policies in the
source data but also the _learning operator_ of the source algorithm. In the
final chapter, we argue that such a model can benefit from the idea of value and
can learn representations that generalizes.

The broader questions remain --- about credit-assignment, generalization, and
the challenges of scale. However, given the history and magnitude of innovations
in this vibrant community of research, we can be sure that revolutionary
developments in the science of RL foundation models are still to come.

// == Dummy <sec:introduction>
// #bibliography("main.bib", style: "american-society-of-civil-engineers") 