
# Introduction : Expectation-Maximization algorithm a.k.a EM algorithm 

## Maximum Likelihood Approaches 
The essence of this study is centered around an elegant, time-honored algorithm first presented in a seminal paper by Arthur Dempster, Nan Laird, and Donald Rubin in 1977. The EM algorithm is commonly employed for determining probability distribution parameters through a maximum likelihood approach by finding maximum likelihood or maximum a posteriori (MAP) estimate, making it an efficient non-Bayesian approach. 

A gentle and short introduction to the algorithm is, according to my taste, given at: http://www.seanborman.com/publications/EM_algorithm.pdf. The EM algorithm has been used in different application domains: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#cite_note-Dempster1977-2. 



## What about Bayesian Approaches 

Pure Neural/Deep Bayesian approaches are intractable. The intractability arises from backpropagating on Neural Networks with millions of decision variables (weights + biases) that we have to deal with at the loss minimization step.

I would like to cite one recent paper working arround the intractability of the task:
```
@article{izmailov_subspace_2019,
  title={Subspace Inference for Bayesian Deep Learning},
  author={Izmailov, Pavel and Maddox, Wesley and Kirichenko, Polina and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
  journal={Uncertainty in Artificial Intelligence (UAI)},
  year={2019}
}
```
## Optimization and complexity analysis aspect 

From an optimization theory point of view, neural networks are trained by minimizing a task using a first-order oracle-based optimizer. However, computing gradients for very large-scale optimization problems, such as those in pure deep Bayesian approaches, can be problematic when trying to obtain a solution with $\epsilon$ precision. The complexity of this task is about $\mathcal{O}(\frac{1}{\epsilon})$ in the smooth case and about $\mathcal{O}(\frac{1}{\epsilon^2})$ for the non-smooth case.

One way to mitigate the aforementioned intractability is to use Free Derivative-based optimizers. There are many types of these optimizers, and some frameworks have a complexity of $\mathcal{O}(\frac{1}{\epsilon^n})$ for some positive integer n.

# Neural-E-M-: Projet motivations and Implementation 
The goal of this project is to implement a General Neural "counterpart" for EM algorithm, which is an efficient maximum likelihood approach especially when combined with a first order oracle type optimizer in case of large scale optimization problem.

To this end, I have identified a relevant paper as an example approach, which I am currently implementing.
This is non official EM implementation of the paper: 
```
@article{DBLP:journals/corr/abs-1708-03498,
  author       = {Klaus Greff and
                  Sjoerd van Steenkiste and
                  J{\"{u}}rgen Schmidhuber},
  title        = {Neural Expectation Maximization},
  journal      = {CoRR},
  volume       = {abs/1708.03498},
  year         = {2017},
  url          = {http://arxiv.org/abs/1708.03498},
  eprinttype    = {arXiv},
  eprint       = {1708.03498},
  timestamp    = {Mon, 13 Aug 2018 16:47:59 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1708-03498.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
This places us in the context of Generative Deep Learning, which is generally known to be more challenging to train, but more suitable in Decision-Making context and generalizes better.

see: 
```
@book{10.5555/1162264,
author = {Bishop, Christopher M.},
title = {Pattern Recognition and Machine Learning (Information Science and Statistics)},
year = {2006},
isbn = {0387310738},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg}
}
```

# Specific part of review of the article  

In the case of RNN-EM, there are K copies of the RNN, each with its own hidden state denoted as $\theta_{k}$. At each timestep, the input to the k-th RNN is $\gamma_{k}(\psi_{k} - x)$, where $\gamma_{k}$ is a scalar parameter (the responsibility terme), $\psi_{k}$ is the k-th vector of the parametric representation of a spatial mixture of $K$ components (interpreted as the mean vector or p Bernoulli parameter), and x is the input data. 

Their is no formal proof about taking this choice of input to the RNNs. One may think about taking only x as input which may be at the first glance intuitive but a poor choice. 

The choice of using the difference between the mean and the input scaled by a factor as input to the RNNs is motivated by the underlying generative model the authors are about to settle. The model assumes that the data x is generated from a mixture of K Normal/Bernoulli distributions with $\psi_{k}$ as the mean and fixed variance/Bernoulli parameter respectively. The scaling factor $\gamma_{k} = P(z_k|x,\psi) = \frac{P(z_{k},x|\psi)}{\sum_{z_{k}}P(z_{k},x|\psi)}$ controls the influence of the mean of each Gaussian/Bernoulli parameter on the input x.

By using $\gamma_{k}(\psi_{k} - x)$ as input to the RNNs, RNN-EM is essentially learning to estimate the parameters of the mixture model by iteratively refining the estimates of the means/Bernoulli parameter and the scaling factors. This allows, normally, the model to better capture the underlying distribution of the data and converge to a more accurate solution. The convergence is not guaranteed though.


The authors say : " ... In order to accurately mimic the M-Step (4) with an RNN, we must impose several restrictions on its weights and structure: the “encoder” must correspond to the Jacobian ...". None of those restrictions are used in this study. 

The use of $\gamma_{k}(\psi_{k} - x)$ instead of $x$ and adding the KL divergence penelization term in the training Loss are good workaround of the aforementioned restrictions and others though.

# Running Process of a Bernoulli Parameterization Version of the Spatial K-Mixture Data Type

To run the RNN-EM training experiment, execute the following command:
```
$> chmod u+x run_experience.sh
$> ./run_experience.sh
```

# Post Scriptum:
- This projects consider:
    - RNN-EM with Bernoulli parameterization for pixels in data, "small detail" we can chose a Noraml version. This choice is related to the data type itself which is binary   
    - Authors static dataset and eventually other static dataset belonging to the state-of-the-art will be considered in the first place.  
- Static dataset of interest is named "shapes.h5" can be found here: https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AAB6WiZzH_mAtCjW6b9okMGea?dl=0.
- Curious about picking up different data sets other than those for clustering purposes
- Feedback is always welcome: my training loss is decreasing, the AMI score is not increasing as fast; Slowly but surely!

