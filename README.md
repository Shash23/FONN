# Deep Learning for Options Trading: Neural Network Price Prediction

Neural Network for Black Scholes Options Pricing

This project loosely implements the **Finance-Informed Neural Network (FINN)** proposed by Prof. Amine Mohamed Aboussalah et. al. with components from other papers. FINN is a hybrid approach to option pricing that integrates ML with finance principles. The model is designed to predict call option prices adhering to the **Black-Scholes Partial Differential Equation** and key **financial boundary conditions**. 

All of the model development and experimentation can be found in the **`development/`** directory. The **`model-dev.ipynb`** notebook serves as a foundation for the core logic and fine-tuning. The finalized model can be found in **`model.py`**. For testing, the project uses manual data generation based on the Black-Scholes model to test the models ability to learn. Future steps include streaming orderbook data. 

This project supports both **GPU and CPU** execution environments. The model will automatically use CUDA if available, falling back to CPU if not. Training times will vary significantly between GPU and CPU execution.

This project remains under **active development**.

## Features

- **Black-Scholes PDE Loss**: Enforces the Black-Scholes partial differential equation as a soft constraint during training
- **Financial Boundary Conditions**: Explicitly incorporates boundary conditions (e.g., payoff at expiry, deep in-the-money behavior) to guide learning.
- **Customizable Neural Network**: Configurable architecture (input dimensions, hidden layers, hidden dimensions) with `SiLU` activation for improved performance.
- **Xavier Normal Initialization**: Robust weight initialization for stable training of deep networks.
- **Gradient Clipping**: Prevents exploding gradients during backpropagation.
- **Data Generation**: Creates Black-Scholes option pricing data for controlled experimentation and validation.
- **Comprehensive Testing (UNDER DEVELOPMENT)**: Includes tests for monotonicity, comparison with Black-Scholes analytical solutions, and error metrics.

## Project Structure

```
option-pricer-neural-net/
├── development/   
│   ├── model-dev.ipynb
│   └── model.py
├── notes/
│   └── notes.pdf
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/option-pricer-neural-net.git
   cd option-pricer-neural-net
   ```

2. Install the required dependencies:
   ```bash
   pip install torch numpy scipy matplotlib ipython
   ```

## Usage

### Training and Testing the Model

The training is done through the `model.py` script. It contains the `train_model()` function for training and `test_model()` to evaluate performance.

1. Navigate to the project root directory.
2. Run the `model.py` script:
   ```bash
   python model.py
   ```
   This script will:
   * Generate a synthetic Black-Scholes dataset.
   * Initialize and train the FINN model.
   * Output training loss per epoch.
   * Compare model predictions against Black-Scholes analytical prices for various test cases, showing percentage error.
   * Perform other, relevant tests inc. monotonicity

### Development Environment

For interactive development, experimentation, and visualization:

1. Open the `model-dev.ipynb` Jupyter notebook located in the `development/` directory.
   ```bash
   jupyter notebook development/model-dev.ipynb
   ```
2. This notebook provides a step-by-step walkthrough of the model's construction, loss function implementation, and training process, allowing for direct modification and observation of results.

## References

This project is mainly inspired by the Finance-Informed Neural Network paper with guidance provided by the other papers and resources listed.

### Core Papers

**[1] The AI Black-Scholes: Finance-Informed Neural Network**  
Amine Mohamed Aboussalah, Xuanze Li, Cheng Chi, Raj Patel (2024). arXiv:2412.12213v1 [cs.LG].  
*This paper introduces the Finance-Informed Neural Network (FINN) framework, which integrates financial principles with deep learning for option pricing.*

**[2] Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations**  
M. Raissi, P. Perdikaris, G. Karniadakis (2019). Journal of Computational Physics 378, 686-707.  
*A foundational paper on Physics-Informed Neural Networks (PINNs), which form the theoretical basis for embedding differential equations into neural network training.*

**[3] The Pricing of Options and Corporate Liabilities**  
F. Black, M. Scholes (1973). Journal of Political Economy 81(3), 637-654.  
*The seminal work introducing the Black-Scholes option pricing model and the associated partial differential equation.*

**[4] Sigmoid-weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning**  
H. Elfwing, E. Uchibe, and K. Doya (2018). arXiv:1811.03378.  
*Introduced the SiLU activation function, which often offers improved performance over ReLU and Tanh.*

### Additional Resources

**Neural Network Architecture Fundamentals**  
[micrograd](https://github.com/karpathy/micrograd) - Karpathy's educational implementation for understanding neural network fundamentals.

**Primary Implementation Reference**  
[AI Black Scholes: Finance-Informed Neural Networks](https://arxiv.org/pdf/2412.12213) - The main paper this project implements.

**Mathematical Testing Methods**  
[Monotonicity Testing for Options](https://www.youtube.com/watch?v=fiCwhsk3PVM) - Reference for testing that call prices increase and exhibit convexity with respect to underlying asset price (S).
