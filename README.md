# Neural-E-M-: Projet and motivations 
The goal of this project is to implement a General Neural "counterpart" for EM algorithm, which is a efficient maximum likelihood approach. 
To this end, I have identified a relevant paper as an example approach, which I am currently implementing.
This is not an official implementation of the paper: https://proceedings.neurips.cc/paper_files/paper/2017/file/d2cd33e9c0236a8c2d8bd3fa91ad3acf-Paper.pdf. 

This places us in the context of Generative Deep Learning, which is generally known to be "more demanding to train" but more suitable in Decision-Making context. (ex. see Bishop:Pattern Recognition and Machine Learning)

# Part of review of the article  

In the case of RNN-EM, there are K copies of the RNN, each with its own hidden state denoted as $\theta_{k}$. At each timestep, the input to the k-th RNN is $\gamma_{k}(\psi_{k} - x)$, where $\gamma_{k}$ is a scalar parameter (the responsibility terme), $\psi_{k}$ is the k-th vector of the parametric representation of a spatial mixture of $K$ components (interpreted as the mean vector or p Bernoulli parameter), and x is the input data. 

Their is no formal proof about taking this choice of input to the RNNs. One may think about taking only x as input which may be at the first glance intuitive but a poor choice. 

The choice of using the difference between the mean and the input scaled by a factor as input to the RNNs is motivated by the underlying generative model the authors are about to settle. The model assumes that the data x is generated from a mixture of K Normal/Bernoulli distributions with $\psi_{k}$ as the mean and fixed variance/Bernoulli parameter respectively. The scaling factor $\gamma_{k} = P(z_k|x,\psi) = \frac{P(z_{k},x|\psi)}{\sum_{z_{k}}P(z_{k},x|\psi)}$ controls the influence of the mean of each Gaussian/Bernoulli parameter on the input x.

By using $\gamma_{k}(\psi_{k} - x)$ as input to the RNNs, RNN-EM is essentially learning to estimate the parameters of the mixture model by iteratively refining the estimates of the means/Bernoulli parameter and the scaling factors. This allows, normally, the model to better capture the underlying distribution of the data and converge to a more accurate solution. The convergence is not guaranteed though.


The authors say : " ... In order to accurately mimic the M-Step (4) with an RNN, we must impose several restrictions on its weights and structure: the “encoder” must correspond to the Jacobian ...". None of those restrictions are used in this study. 

The use of $\gamma_{k}(\psi_{k} - x)$ instead of $x$ and adding the KL divergence penelization term in the training Loss are good workaround of the aforementioned restrictions and others though.

# Running Bernoulli parameterization of the spatial mixture experiment

To run the RNN-EM training experiment, execute the following command (en mode "spectateur"):
```
$> chmod u+x run_experience.sh
$> ./run_experience.sh
```
# Post Scriptum:
- This projects consider:
    - RNN-EM with Bernoulli parameterization for pixels in data, "small detail" 
    - Authors static data and eventually other static data belonging to the state-of-the-art will be considered in the first place.  
- Dataset of interest is named "shapes.h5" can be found here: https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AAB6WiZzH_mAtCjW6b9okMGea?dl=0.
- (curious about picking up different data sets other than those for clustering purposes)
- Feedback is always welcome: my training loss is decreasing, the AMI score is not increasing as fast. Slowly but surely!

