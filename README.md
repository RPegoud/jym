# **JYM (Jax Gym)**

<h1 align="center">
  <a>
    <img src="https://github.com/RPegoud/jym/assets/60786847/f7939998-dacc-4ca5-acf2-56a0dc018377" width="400" /></a><br>
  <b>JAX implementations of standard RL algorithms and vectorized environments</b><br>
</h1>

<p align="center">
<a href="https://github.com/RPegoud/jym/issues">
  <img src="https://img.shields.io/github/issues/RPegoud/jym" /></a>
<a href="https://github.com/RPegoud/jym/actions/workflows/lint_and_test.yaml">
        <img src="https://github.com/RPegoud/jym/actions/workflows/lint_and_test.yaml/badge.svg" /></a>
        <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
        <a href="https://github.com/RPegoud/jym/blob/main/LICENSE">
  <img src="https://img.shields.io/github/license/RPegoud/jym" /></a>

</p>

<table>
  <tr>
    <th colspan="5" align="left">🚀 Stack:</th>
  </tr>
  <tr>
    <td align="left">
      <a target="blank">
        <img align="center" src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/267_Python_logo-512.png" alt="Python" height="40" width="40" />
        Python
      </a>
    </td>
    <td align="left">
      <a >
        <img align="center" href="https://jax.readthedocs.io/en/latest/index.html" target="blank" src="https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg" alt="JAX" height="40" width="40" />
        JAX
      </a>
    </td>
    <td align="left">
      <a target="blank">
        <img align="center" href="https://dm-haiku.readthedocs.io/en/latest/" src="https://avatars.githubusercontent.com/u/144367226?s=280&v=4" alt="Haiku" height="40" width="40" />
        Haiku
      </a>
    </td>
    <!-- <td align="left">
      <a  target="blank">
        <img align="center" href="https://flax.readthedocs.io/en/latest/" src="https://raw.githubusercontent.com/google/flax/main/images/flax_logo_250px.png" alt="Flax" height="40" width="40" />
        Flax
      </a>
    </td> -->
    <td align="left">
      <a target="blank">
        <img align="center"  href="https://optax.readthedocs.io/en/latest/" src="https://optax.readthedocs.io/en/latest/_static/logo.svg" alt="Optax" height="40" width="40" />
        Optax
      </a>
    </td>
  </tr>
</table>

## 🌟 ***Key Features***

* 🐍  **Clean** and **beginner-friendly** implementation of **Reinforcement Learning algorithms** in **JAX**
* ⚡ **Vectorized environments** for lightning-fast training
* 👩‍👨‍👦‍👧 **Parallel agent training** for statistically significant results
* 📊 **Plotly graphs** enabling state value visualization throughout training and averaged performance reports
* ✅ **Easy installation** using **Poetry** virtual environments
* ✍️ **Code walkthroughs**:
  * [***Vectorize and Parallelize RL Environments with JAX: Q-learning at the Speed of Light⚡***](https://towardsdatascience.com/vectorize-and-parallelize-rl-environments-with-jax-q-learning-at-the-speed-of-light-49d07373adf5), published in ***Towards Data Science***
  * [***A Gentle Introduction to Deep Reinforcement Learning in JAX 🕹️***](https://towardsdatascience.com/a-gentle-introduction-to-deep-reinforcement-learning-in-jax-c1e45a179b92), published in ***Towards Data Science***, selected as part of the ***"Getting Started"*** column

## ✅ ***Progress***

### 🤖 Algorithms

| Type    | Name                                      | Source                  |
| ------- | ----------------------------------------- | ----------------------- |
| Bandits | [Simple Bandits ($\epsilon$-Greedy policy)](https://github.com/RPegoud/jym/blob/main/jym/agents/simple_bandit.py) | Sutton & Barto, 1998    |
| Tabular | [Q-learning](https://github.com/RPegoud/jym/blob/main/jym/agents/q_learning.py)                                | Watkins & Dayan, 1992   |
| Tabular | [Expected SARSA](https://github.com/RPegoud/jym/blob/main/jym/agents/expected_sarsa.py)                           | Van Seijen et al., 2009 |
| Tabular | [Double Q-learning](https://github.com/RPegoud/jym/blob/main/jym/agents/double_q_learning.py)                         | Van Hasselt, 2010       |
| Deep RL | [Deep Q-Network (DQN)](https://github.com/RPegoud/jym/blob/main/jym/agents/dqn.py)                      | Mnih et al., 2015       |

### 🌍 Environments

| Type               | Name                     | Source                          |
| ------------------ | ------------------------ | ------------------------------- |
| Bandits            | [Casino (K-armed Bandits)](https://github.com/RPegoud/jym/blob/main/jym/envs/bandits/k_armed_bandits.py) | Sutton & Barto, 1998            |
| Tabular            | [GridWorld](https://github.com/RPegoud/jym/blob/main/jym/envs/grids/grid_world.py)               | -                               |
| Tabular            | [Cliff Walking](https://github.com/RPegoud/jym/blob/main/jym/envs/grids/cliff_walking.py)            | -                               |
| Continuous Control | [CartPole](https://github.com/RPegoud/jym/blob/main/jym/envs/control/cartpole.py)                 | Barto, Sutton, & Anderson, 1983 |
| MinAtar            | [Breakout](https://github.com/RPegoud/jym/blob/main/jym/envs/minatar/breakout.py)                 | Young et al., 2019              |

## ⌛ ***Coming Soon***

### 🤖 Algorithms

| Type    | Name                                       |
| ------- | -----------------------------------------  |
| Bandits              | UCB (Upper Confidence Bound) |
| Tabular (model based)| Dyna-Q, Dyna-Q+ |

### 🌍 Environments

| Type               | Name                     |
| ------------------ | ------------------------ |
| MinAtar| Asterix, Freeway, Seaquest, SpaceInvaders |

## 🧪 ***Experiments***

* ### **K-armed Bandits Testbed**

<a href="https://github.com/RPegoud/jax_rl/blob/main/notebooks/bandits/incremental_bandits.ipynb">
 <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
</a>

  Reproduction of the 10-armed Testbed experiment presented in [***Reinforcement Learning: An Introduction***](http://incompleteideas.net/book/the-book-2nd.html) (chapter 2.3, page 28-29).

  This experiment showcases the difference in performance between different values of epsilon and therefore the long-term tradeoff between exploration and exploitation.
  
<div align="center">
  <table>
    <tr>
      <td>
        <img src="https://raw.githubusercontent.com/RPegoud/jax_rl/main/images/10-armed%20bandits%20testbed.png" alt="Description for first image" width="400"/>
        <p align="center">10-armed Testebed environment</p>
      </td>
      <td>
        <a href="https://plotly.com/~Ryan_pgd/7/?share_key=XO3yJyF8asLycAUymU4pbS" target="_blank" title="K-armed Bandits Testbed distribution"><img src="https://plotly.com/~Ryan_pgd/7.png?share_key=XO3yJyF8asLycAUymU4pbS" alt="K-armed Bandits Testbed distribution" width="415"/></a>
        <p align="center">K-armed Bandits JAX environment</p>
      </td>
    </tr>
  </table>

</div>

<div align="center">
<table>
  <tr>
    <td>
      <img src="https://miro.medium.com/v2/resize:fit:1400/1*n5up95W-Zy5gC0Momy7LaQ.png" alt="Description for first image" width="430"/>
      <p align="center">Results obtained in Reinforcement Learning: An Introduction</p>
    </td>
    <td>
      <a href="https://plotly.com/~Ryan_pgd/9/?share_key=B9eEph84sA0EoX3XL7eFhs" target="_blank" title="10-Armed bandits testbed reproduction"><img src="https://plotly.com/~Ryan_pgd/9.png?share_key=B9eEph84sA0EoX3XL7eFhs" alt="K-armed Bandits Testbed" width="400"/></a>
    <p align="center">Replicated results using the K-armed Bandits JAX environment</p>
    </td>
  </tr>
</table>
</div>

* ### **Cliff Walking**

<a href="https://github.com/RPegoud/jym/blob/main/notebooks/tabular/cliff_walking/q_learning_cliff_walking.ipynbb">
 <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
</a>

Reproduction of the CliffWalking environment presented in Reinforcement Learning: An Introduction (chapter 6, page 132).

This experiment highlights the difference in behavior between TD algorithms, Q-learning being greedy (as the td target is the maximum Q-value over the next state) and Expected Sarsa being safer (td target: expected Q-value over the next state).

<div align="center">
<table>
  <tr>
    <td>
      <img src="https://live.staticflickr.com/65535/49199511478_3054654b30.jpg" width="430"/>
      <p align="center">Described behaviour for the CliffWalking environment</p>
    </td>
    <td>
      <a><img src="https://raw.githubusercontent.com/RPegoud/jym/main/images/Cliff_walking.jpg" alt="" width="400"/></a>
    <p align="center">Comparison of Expected Sarsa (top) and Q-learning (bottom) on CliffWalking</p>
    </td>
  </tr>
</table>
</div>

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

## 📝 References

* [***Reinforcement Learning: An Introduction***](http://incompleteideas.net/book/the-book-2nd.html) Sutton, R. S., & Barto, A. G., The MIT Press., 2018
* [***Writing an RL Environment in JAX***](https://medium.com/@ngoodger_7766/writing-an-rl-environment-in-jax-9f74338898ba), Nikolaj Goodger, Medium, Nov 14, 2021
* [***JAX Tutorial Playlist***](https://www.youtube.com/watch?v=SstuvS-tVc0&list=PLBoQnSflObckOARbMK9Lt98Id0AKcZurq), Aleksa Gordić - The AI Epiphany, YouTube, 2022

<br/>

| ![](https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg) |
|:--:|
| [***Official JAX Documenation***](https://jax.readthedocs.io/en/latest/index.html) |
