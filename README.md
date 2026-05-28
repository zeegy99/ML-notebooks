I will document the ML papers I implement for enjoyment and challenge, alongside some random applications!

** [Diffusion Models]**

We start with [DDPM] (https://arxiv.org/pdf/2006.11239) since this is the stable of all diffusion models and I wanted to know how people generate images with AI.

Basic Background: The Diffusion Probabilistic Model utilizes Markov Chains and some elegant properties to generate images from Random Noise. \

The core idea is that given an image that is N(0, 1) Gaussian Noise, we slowly denoise through a backwards process until we end up with $x_0$ or our 'initial image'.\

For our Backwards Process, we define 

**$$p_\theta(x_{t-1}|x_t) = N(x_{t-1}; \mu_\theta(x_t, t), \sum_\theta(x_t, t))$$.**  

Essentially, our reverse process takes our current state x_t and removes some Gaussian Noise.

For More information and all of the math behind this paper, look [Here] (https://www.overleaf.com/read/pszbcvbmptms#9f695c) 



