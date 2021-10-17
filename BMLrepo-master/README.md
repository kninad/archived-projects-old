## BMLrepo (Bayesian ML)

### Project Idea

**Online variational Bayesian inference, also called "stochastic variational inference" (SVI)**

Variational inference is a general framework for doing approximate posterior inference for Bayesian models (here is a recent survey on the topic: https://arxiv.org/pdf/1601.00670.pdf). SVI is an "online" way of doing variational inference. It enables scaling up Bayesian models to massive data sets and is very similar to stochastic gradient based methods used for online learning. There is a good survey paper on this topic (http://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) published in 2013. Since then, there has been a lot of activity on this topic. Some possible projects:

(1) Survey this topic in more detail, focusing especially on some recent advances and maybe re-implement it for some Bayesian models for which it has already been tried.

(2) Work out the details of SVI for some Bayesian model on which SVI has not been tried yet (some suggestions)

(3) A combination of both (1) and (2).

---

**Other advances**

Another interesting project on variational inference would be to look at more advanced variational inference methods that address some of the limitations of variational inference (e.g., fully factorized assumption, handling non-conjugate distributions, distributed variational inference, etc.), thinking of improving these and/or using it to do inference for some Bayesian model. 

Another possibility could to to do an extensive survey of these (and other related methods not listed above), and maybe also do an experimental comparison of some of these on some problems. Some relevant papers are (list not exhaustive): 

- Black Box Variational Inference (AISTATS 2014): http://www.cs.columbia.edu/~blei/papers/RanganathGerrishBlei2014.pdf 

- Hierarchical Variational Models (ICML 2016): http://www.jmlr.org/proceedings/papers/v48/ranganath16.pdf 

- Boosting Variational Inference (2016): https://arxiv.org/pdf/1611.05559.pdf 

- Variational Boosting: Iteratively Refining Posterior Approximations (2016): https://arxiv.org/pdf/1611.06585.pdf 

- Automatic Variational Inference in Stan (NIPS 2015): http://papers.nips.cc/paper/5758-automatic-variational-inference-in-stan.pdf 

- Variational Inference with Normalizing Flows (ICML 2015): http://jmlr.org/proceedings/papers/v37/rezende15.pdf 

- Variational Consensus Monte Carlo (NIPS 2016): https://papers.nips.cc/paper/5888-variational-consensus-monte-carlo.pdf 

- Stochastic Expectation Propagation (NIPS 2015): http://papers.nips.cc/paper/5760-stochastic-expectation-propagation.pdf (note: EP is different from variational inference but the underlying idea is similar -- they both turn an inference problem into an optimization problem of finding the "closest" distribution)

- Streaming Variational Bayes (NIPS 2013): https://papers.nips.cc/paper/4980-streaming-variational-bayes.pdf (note: a way to do online variational inference when you don't know the data set size in advance)

