# options pricing neural network

summary of the paper

i. input/output specifications

- simulated asset prices S (eg. generated from GBM/Heston Models)
- Strike price K and time to maturity T
- Derived Features

=>
 
- predicted european call/put option price C(S, K, T)


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

    



