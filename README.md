I will document the ML papers I implement for enjoyment and challenge, alongside some random applications!

**Diffusion Models**

We start with [DDPM] (https://arxiv.org/pdf/2006.11239) since this is the stable of all diffusion models and I wanted to know how people generate images with AI.

Basic Background: The Diffusion Probabilistic Model utilizes Markov Chains and some elegant properties to generate images from Random Noise. \

The core idea is that given an image that is N(0, 1) Gaussian Noise, we slowly denoise through a backwards process until we end up with $x_0$ or our 'initial image'.\


For More information and all of the math behind this paper, look [Here] (https://www.overleaf.com/read/pszbcvbmptms#9f695c) 



**Neural Networks / Backpropogation**

A simple Neural Network Trained to recognize MNIST dataset.
\\

1. Forward Propogation: This takes your inputs (X) and spits out some output ($\hat{y}_i$). The way we will do this is we have 3 layers. We go from the input layer (784 neurons) $\to$ Hidden Layer 1 (10 neurons) $\to$ Output Layer (10 Neurons). \\


    When going from input layer to Hidden layer 1, we will have some W (weights) with shape (10 x 784) multiplied with x (784 x n). What this means is that for each of the 784 features taken from X, there will be some combination of those weights that determine how many of our 10 neurons are activated. Then we use some non-linear activation function (ReLu in this case), otherwise our entire NN will be nested linear functions which is linear. \\
    

    Note: Consider f(g(x)). If f and g are both linear functions like mx + b, then f(g(x)) is also of the form mx + b which will not capture the complexities of our questions. \\

    When going from Hidden Layer 1 to Output layer, we apply softmax which turns each output into a probability, with $\sum neurons$ = 1. \\

2. Back Prop: We are comparing the answer generated from forward prop to the real answer and go back and adjust the weights using gradient descent.

    We want to turn our answer vector into similar form as our output, which means we will use one-hot. This turns y which has value 8 into a vector [0, 0, 0, 0, 0, 0, 0, 0, 1, 0].T 

    For our first loss function from softmax, we will use the cross-entropy-loss function defined as \[ L = -\sum (y_i log(\hat{y}_i)) \]


    Then we apply gradient descent which moves these values from (learning rate) from where they are. Over ITERATIONS, these weights will become appropriately adjusted. 


3. Math:
    1. For our softmax function (dW2), consider $\frac{dL}{dW2} = \frac{dL}{d\hat{y}_i} \cdot \frac{d\hat{y}_i}{dz_2} \cdot \frac{dz_2}{dW2}$. The reason why we need all of these partials is because we can rewrite L in terms of the next terms. \\

        To solve this, lets look at \[ L := -\sum (y_i log(\hat{y}_i)) \] and break it down into its components. Here, $y_i$ is predetermined as 1 or 0, and we know that $\hat{y}_i$ is our softmax function from above, $\frac{e^{x_i}}{\sum e^{x_j}}$. Thus, our first partial comes out to $\frac{-y_i}{\hat{y}_i}$. \\

        For our next partial, (Softmax), we must consider this into 2 cases. On the top we have some $e^{x_i}$ term. So consider the case where our partial is with respect ti $x_i$ and then with respect to $x_j$ when $j \neq i$. \\

        In case 1: We will use simple calculus to solve this partial. Observe that \[
        \hat{y}_i := \frac{e^{x_i}}{\sum e^{x_j}}
        \]. Thus, we can simplify our expression down to $\hat{y}_i (1 - \hat{y}_i)$. \\

        Then, for the case when $i \neq j$, we will get $-\hat{y}_i \hat{y}_j$. \\

        Finally, for our last term $\frac{dz_2}{dW2}$, remember that Z2 = W2 $\cdot$ A1 (previous output layer) + $b_2$. So when taking the partial we are just left with $A1$. \\

        Putting this all together, we get the desired result. 

    \\

    Similarly for (b), just replace the last fraction with $dB2$. In fact, the partial comes out to be (1), which is even simpler! \\

    For our next term (dz1), Apply the same process. I'll write out the fraction decomposition and you can work through the math again (or re-use from above). $\frac{dL}{dZ1} = \frac{dL}{dA1}\frac{dA1}{dZ1}$. Then rewrite $\frac{dL}{dA1} := \frac{dL}{dZ2} \frac{dZ2}{dA1}$. We have already computed these values. The only term that is somewhat confusing is $\frac{dA1}{dz1}$. How do you take the derivative of a ReLu function? Well, since ReLu is 0 when x < 0, the derivative is 0 for x < 0. Similarly, since it is x when x $\geq$ 0, the derivative is 1 for all x > 0. Thus, we can use a trick and write $Z > 0$ since True gets assigned to 1. 
