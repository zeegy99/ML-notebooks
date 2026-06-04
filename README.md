I will document the ML papers I implement for enjoyment and challenge, alongside some random applications!

## ML Fundamentals

We start with a classification question and work through EDA and build some features to determine if someone is risky or not (to give a loan to)
[EDA](eda.ipynb)
[Logistic Regression](ml.ipynb)


## Diffusion Models

We start with [DDPM](https://arxiv.org/pdf/2006.11239): [Link to Notebook](DDPM.ipynb) since this is the stable of all diffusion models and I wanted to know how people generate images with AI.

Basic Background: The Diffusion Probabilistic Model utilizes Markov Chains and some elegant properties to generate images from Random Noise. \

The core idea is that given an image that is N(0, 1) Gaussian Noise, we slowly denoise through a backwards process until we end up with $x_0$ or our 'initial image'.\


For More information and all of the math behind this paper, look [Here] (https://www.overleaf.com/read/pszbcvbmptms#9f695c) 



**Neural Networks / Backpropogation**

[Link to Notebook](mnist_noteobok.ipynb)


