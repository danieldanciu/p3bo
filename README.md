# The problem
Welcome to your onsite day at Cradle, Gino!

Your mission, should you choose to accept it, is to ... (drum rolls) implement the P3BO algorithm! The algorithm
is described on page 4 of the [P3BO paper](https://arxiv.org/pdf/2006.03227.pdf). You don't need 
to implement the adaptive version.

Most of the code that you write will go into [p3bo.py](p3bo.py). In order to help test your implementation we  
provide 3 optimizers to run black box optimization on:

- [Adalead](flexs/optimizers/adalead.py)
- [Genetic](flexs/optimizers/genetic_algorithm.py)
- [Random]((flexs/optimizers/random.py))

Each of these optimizers (also called Explorers) exposes a method `propose_sequences()` (the equivalent of `A.propose()`
in P3BO) and a method called `fit()` (the equivalent of `A.fit()` in P3BO).

The model (or landscape or oracle) that you are optimizing against is going to be a fake one (a mock) so that we avoid
running costly/hard-to-setup models. The fake oracle is just a noisy Levensthein (a.k.a. edit) distance from the target sequence that
is considered to be the unique global optimum solution (so the further you are from the optimum the higher the value,
which means that you have a minimization problem). The noisy oracle is implemented in
[noisy_abstract_model.py](flexs/models/noisy_abstract_model.py) (which is a class that adds noise to a ground truth
function proportional to the distance from an already measured sequence). The ground truth that is corrupted is
implemented in [levensthein.py](flexs/models/levensthein.py) and is simply the edit distance to the target global
optimum.

Your goal is to implement P3BO and to use the 3 optimizers (Adalead, Genetic, Random) to get as close as possible to the
global optimum, using the noisy oracle. The optimizers are all seeded with a sequence that is x% identical to the
optimum value. I have no idea what a reasonable value for x is. I'd start with 95%, and then go lower or higher if this
converges too quickly or doesn't converge.

The task is subdivided into multiple smaller tasks:

* implement `get_starting_sequence`, a function that returns a sequence that is x% identical to a given sequence, where
  x is a number from 0 to 100 (0 means all positions may differ, 100 means all positions are identical)
* implement p3bo itself
* plot some interesting results (fitness value over time, which optimizer performs best/has the most values in
* the population, whatever other interesting metrics you can think of)

All this code has never been tested together and has probably bugs/glitches, particularly at the interface level
(we don't expect the algorithm implementations thmeselves to have any bugs).

Have fun!

# Installation instructions
1. Clone the git repository:
```bash
git clone https://github.com/danieldanciu/p3bo.git
```
2. Create a conda environment (but you can also use poetry or pip, whatever you prefer):
```bash
conda create --name p3bo python=3.8
conda activate p3bo
```

3. Install dependencies
```bash
cd p3bo

```
