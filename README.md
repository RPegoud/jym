# **JAX-RL** <img src='https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg' alt="Environment" width="50" />

This repository contains JAX implementations of several RL algorithms and environments.

## 🌟 ***Key Features***

* 🐍  **Clean** and **beginner-friendly** implementation of **Reinforcement Learning algorithms** in **JAX**
* ⚡ **Vectorized environments** for lightning-fast training
* 👩‍👨‍👦‍👧 **Parallel agent training** for statistically significant results
* 📊 **Plotly graphs** enabling state value visualization throughout training and averaged performance reports
* ✅ **Easy installation** using **Poetry** virtual environments
* ✍️ **Code walkthroughs**:
  * [***Vectorize and Parallelize RL Environments with JAX: Q-learning at the Speed of Light⚡***](https://towardsdatascience.com/vectorize-and-parallelize-rl-environments-with-jax-q-learning-at-the-speed-of-light-49d07373adf5), published in ***Towards Data Science***

## ✅ ***Progress***

* 🤖 Algorithms:
  * Bandits
    * Simple Bandits ($\epsilon$-Greedy policy)
  * Tabular
    * Q-learning
    * Expected SARSA
    * Double Q-learning
* 🌍 Environments:
  * Bandits
    * Casino *(fancy name for K-armed Bandits)*
  * Tabular
    * GridWorld

## ⌛ ***Coming Soon***

* 🤖 Algorithms:
  * Bandits:
    * UCB (Upper Confidence Bound)
  * Tabular:
    * Dyna-Q, Dyna-Q+
  * Deep RL:
    * Deep Q Network (DQN)
* 🌍 Environments:
  * Cliff Walking
  * Frozen Lake
  * CartPole

<!-- ## 🚀 ***Performances***

| Algorithm  | Type        | Environment | Updates per step | Steps/second (single CPU) | Number of parallel agents | Number of training steps | Runtime (s) |
|:---------- | ----------- |:----------- | ---------------- |:------------------------- | ------------------------- | ------------------------ | ----------- |
| Q-learning | Model-free  | GridWorld   | 1                | 1.500.000                 | 30                        | 100.000                  | 2           |
| Dyna-Q     | Model-based |             |                  | *coming soon*             |                           |                          |             |
| Dyna-Q+    | Model-based |             |                  | *coming soon*             |                           |                          |             | -->

## 💾 Installation

To install and set up the project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/RPegoud/jax_rl.git
   ```

2. Navigate to the project directory:

   ```bash
   cd jax_rl
   ```

3. Install Poetry (if not already installed):

   ```bash
   python -m pip install poetry
   ```

4. Install project dependencies using Poetry:

   ```bash
   poetry install
   ```

5. Activate the virtual environment created by Poetry:

   ```bash
   poetry shell
   ```

## Use

1. Modify the main.py file depending on the test you want to perform and run:

   ```bash
   python main.py

## 📝 References

* [***Reinforcement Learning: An Introduction***](http://incompleteideas.net/book/the-book-2nd.html) Sutton, R. S., & Barto, A. G., The MIT Press., 2018
* [***Writing an RL Environment in JAX***](https://medium.com/@ngoodger_7766/writing-an-rl-environment-in-jax-9f74338898ba), Nikolaj Goodger, Medium, Nov 14, 2021
* [***JAX Tutorial Playlist***](https://www.youtube.com/watch?v=SstuvS-tVc0&list=PLBoQnSflObckOARbMK9Lt98Id0AKcZurq), Aleksa Gordić - The AI Epiphany, YouTube, 2022

<br/>

| ![](https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg) |
|:--:|
| [***Official JAX Documenation***](https://jax.readthedocs.io/en/latest/index.html) |
