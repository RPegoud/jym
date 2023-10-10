# **JAX-RL**

This repository contains JAX implementations of several RL algorithms and environments.

## 🌟 ***Key Features***

* 🐍 <img src='https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg'
  alt="Environment" width="25" /> Clean and beginner-friendly implementation of RL algorithms in JAX
* ⚡ Vectorized environments for lightning fast training
* 🤖 Implemented Algorithms: Q-Learning, *(coming soon: DQN, DDQN, Dyna-Q, Dyna-Q+)*
* 📊 Plotly graphs enabling state value visualization throughout training and averaged performance reports
* ✅ Easy installation using Poetry virtual environments
* ✍️ All implementations will be detailed step by step [here](https://machine-learning-blog.vercel.app)

## ✅ ***Progress***

* 🤖 Algorithms:
  * Q-learning
* 🌍 Environments:
  * GridWorld

## ⌛ ***Coming Soon***

* 🤖 Algorithms:
  * DQN
  * Dyna-Q, Dyna-Q+
* 🌍 Environments:
  * CartPole

## 🚀 ***Performances***

| Algorithm  | Type        | Updates per step     | Runtime  |
|:---------- | ----------- | -------------------- |:---------------------------------- |
| Q-learning | Model-free  | 1                    | *coming soon*                      |
| Dyna-Q     | Model-based | 101                  | *coming soon*                      |
| Dyna-Q+    | Model-based | 101                  | *coming soon*                      |

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

* [***Writing an RL Environment in JAX***](https://medium.com/@ngoodger_7766/writing-an-rl-environment-in-jax-9f74338898ba), Nikolaj Goodger, Medium, Nov 14, 2021
* [***JAX Tutorial Playlist***](https://www.youtube.com/watch?v=SstuvS-tVc0&list=PLBoQnSflObckOARbMK9Lt98Id0AKcZurq), Aleksa Gordić - The AI Epiphany, YouTube, 2022

<br/>

| ![](https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg) |
|:--:|
| [***Official JAX Documenation***](https://jax.readthedocs.io/en/latest/index.html) |
