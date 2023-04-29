# Neural-E-M-:
The goal of this project is to implement the general EM algorithm, which is a highly efficient maximum likelihood approach.
A relevent paper as an approach example is being implemented:
This is a non-official implementation of the paper: https://proceedings.neurips.cc/paper_files/paper/2017/file/d2cd33e9c0236a8c2d8bd3fa91ad3acf-Paper.pdf. 

# Part of review of the article : 

In the case of RNN-EM, there are K copies of the RNN, each with its own hidden state denoted as $\theta_{k}$. At each timestep, the input to the k-th RNN is $\gamma_{k}(\psi_{k} - x)$, where $\gamma_{k}$ is a scalar parameter (the responsibility terme), $\psi_{k}$ is the k-th vector of the parametric representation of a spatial mixture of $K$ components (interpreted as the mean vector), and x is the input data. 

Their is no formal proof about taking this choice of input to the RNNs. One may think about taking only x as input which may be at the first glance intuitive but a poor choice. 

The choice of using the difference between the mean and the input scaled by a factor as input to the RNNs is motivated by the underlying generative model the authors are about to settle. The model assumes that the data x is generated from a mixture of K Gaussian distributions with means $\psi_{k}$ and fixed variance. The scaling factor $\gamma_{k}$ controls the influence of the mean of each Gaussian on the input x.

By using $\gamma_{k}(\psi_{k} - x)$ as input to the RNNs, RNN-EM is essentially learning to estimate the parameters of the mixture model by iteratively refining the estimates of the means and the scaling factors. This allows the model to better capture the underlying distribution of the data and converge to a more accurate solution. The convergence is not guaranteed though.


The authors say : " ... In order to accurately mimic the M-Step (4) with an RNN, we must impose several restrictions on its weights and structure: the “encoder” must correspond to the Jacobian ...". None of those restrictions are used in this study. 

The use of $\gamma_{k}(\psi_{k} - x)$ instead of $x$ and adding the KL divergence penelization term in the training Loss are good workaround of the aforementioned restrictions and others though.


# Post Scriptum:
- The dataset can be found here: https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AAB6WiZzH_mAtCjW6b9okMGea?dl=0."
- Further steps of this projects will consider other (static) state-of-the-art data set. 
- The trainer for static case is comming as soon as possible. 
- Feedbacks are always welcomed!  


