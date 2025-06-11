# options pricing neural network

repo is currently in progress

access model_dev.ipynb for research

## High Level Model Implementation

i. input/output specifications

- simulated asset prices S (eg. generated from GBM/Heston Models)
- Strike price K and time to maturity T
- Derived Features

=>
 
- predicted call/put option price C(S, K, T)


ii. network design
    
    a. layers
    - fully connected forward feed network with 2-3 hidden layers (dropout potential)
    - Input layer: normalized S, K, T
    - Hidden Layers: 64-128 Neurons
    - Output Layer: Scalar output for predicted price

    b. activation
    - ReLU or tanh
    - Final Layer is Linear

    c. how pde informs structure
    - Black Scholes PDE is embedded into the training loss
    - uses autograd to compute partial derivates
    - derivatives plugged into PDE residual to enforce consistency with Black-Scholes
    
iii. data generation/collection approach

- Simulated asset paths using stochastic processes
    - Geometric Brownian Motion
    - Heston Model (stochastic volatility)

    - Generate a grid of spot prices, maturity, and strikes to create synthetic option prices(BS formula or Monte Carlo)
    
    - Real market data for testing



iv. loss terms and training schedules

    a. Loss terms
        - Supervised Loss
        - PDE Loss
        - Total Loss

    b. Training Schedule
        - Optimizer: Adam
        - Epochs: 1000+ depending on convergence (will be smaller for my project)
        - Learning Rate: ~0.001 with decay
        - Batch Size: Variable often 64-256
        - Early Stopping based on validation loss

https://docs.pytorch.org/docs/stable/generated/torch.ones_like.html

ones_like is mad unserious but lowk cold:
torch.ones_like(input) = torch.ones(input.size(), dtype=input.dtype, layout=input.layout device=input.device)

## Materials of Reference

Refresher on Neural Net Architecture
micrograd
https://github.com/karpathy/micrograd

This is the paper I am loosely implementing
AI Black Scholes: Finance-Informed NN
https://arxiv.org/pdf/2412.12213

** `Math Fundamentals` **

Test for Monotincity
https://www.youtube.com/watch?v=fiCwhsk3PVM
- This is for testing that call prices SHOULD increase and be convex with S








