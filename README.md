# Statistical Machine Learning (SF2957)

This repository contains coursework, comprehensive reports, and computational implementations for the Statistical Machine Learning course. The codebase is divided into two major projects, moving from image classification techniques to reinforcement learning environments.

---

## Project 1: Multi-Class Classification on MNIST (`/Project_1_MNIST_Classification`)
This project explores different classification algorithms applied to the MNIST handwritten digits dataset, evaluating performance, model complexity, and data requirements.

### Exercises & Implementations
* **1. Support Vector Machines (SVM):** Implemented a linear SVM from scratch using stochastic subgradient descent. The code evaluates the effect of training data size and analyzes convergence behavior using different learning rate decay schedules (alpha) to minimize the empirical hinge loss.
* **2a. Gaussian Process Classification (GPC):** Applied GPC leveraging `scikit-learn` across four distinct kernels (RBF, DotProduct, Matern, Rational Quadratic). The exercise highlights the curse of dimensionality, showing how specific kernels (like RBF) struggle in high-dimensional image spaces compared to linear or multi-scale kernels.
* **2b. Gaussian Process Regression (GPR):** Reformulated the multi-class classification task into a regression problem using one-hot encoded targets. Demonstrated that the closed-form Gaussian posterior of GPR yields a highly robust, less kernel-sensitive alternative to GPC, especially effective in limited data regimes.
* **3a. Baseline Convolutional Neural Network (CNN):** Constructed a baseline CNN using `PyTorch`, comprising two convolutional layers and two fully connected layers with ReLU activations, achieving a near-perfect test accuracy.
* **3b. CNN Architecture Optimization:** Conducted a systematic complexity reduction by introducing max pooling and iteratively reducing the number of output channels. The exercise maps the direct trade-off between model parameter count and test error, achieving excellent accuracy while reducing the model footprint by over 96%.

---

## Project 2: Reinforcement Learning for Blackjack (`/Project_2_RL_Blackjack`)
This project builds a reinforcement learning framework to derive and evaluate optimal playing policies for Blackjack under varying deck constraints.

### Exercises & Implementations
* **Environment Engineering:** Utilized custom OpenAI `gymnasium` environments to model the game across two distinct state spaces: an **extended representation** (tracking the exact count of each card value held) and a **sum-based representation** (tracking player sum, dealer showing card, and usable ace status).
* **On-policy Monte Carlo Control:** Implemented an MC algorithm to estimate the action-value function by averaging returns at the end of each episodic hand. The experiments demonstrate the high-variance, slower-convergence nature of end-of-episode updates.
* **Q-Learning (Off-policy TD Control):** Built a Q-learning agent leveraging step-by-step temporal-difference updates. The implementation benchmarks learning efficiency across both state spaces, proving that the reduced sum-based space converges significantly faster due to higher visit frequencies per state.
* **Hyperparameter Tuning:** * Tuned the learning rate step-size schedule by experimenting with the polynomial decay parameter (omega) to balance rapid adaptation with value stability.
  * Compared constant versus decaying exploration rates (epsilon-greedy), demonstrating that decaying exploration yields higher asymptotic average rewards.
* **Optimal Policy Extraction:** Exported the final converged Q-values to construct comprehensive "Basic Strategy" action tables, mapping the optimal Hit/Stick decisions for both Hard and Soft hand totals against the dealer's visible card.