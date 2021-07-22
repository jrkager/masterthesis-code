# Computer-Assisted Proofs for Combinatorial Feasi- bility Problems via Mixed Integer Programming

Code used in my masterthesis at TUM, Faculty of Mathematics, under superivision of Prof. Weltge.

## Installation

The code uses Python 3.8 and Gurobi 9.1. Additional packages used are ```jsonpickle``` and ```pylgl```. A conda specification file is found under ```requirements.txt``` and can be installed via:

```conda create -n makager --file requirements.txt```

## Simplex

The script is started by running the ```main.py``` file from command line, using these optional arguments:

- ```-d```: dimension, as integer
- ```-s```: shape, as string, one of ```simpl``` (pyramid), ```simplflip``` (flipped pyramid), ```box```, ```dyn``` (dynamic growing using a flipped pyramid as standard shape).
- ```-c```: maximum cardinality, as integer
- ```-e```: maximum extent, as integer
- ```-v```: verbosity, as integer from 0 to 4
- ```--dbpath```: path of database (default set to ```./savefiles/database-master.json```)
- ```--delete```: delete the chosen savefile from the database (list of savefiles will appear and user can enter a choice)

## Hypercube

The code is organized in three files:

- ```bestepsilon_original.py```: Implementing the test-set-generation and the LP to find the best $\varepsilon$ to use for a specific dimension. The ```perm``` and ```flip``` reductions can optionally be disabled.
- ```bestepsilon_memeff.py```: Same as above, but using the memory efficient version of the test-set-generation.
- ```slicing_model.py```: Finding the smallest number of hyperplanes needed to slice the hypercube of specific dimension by (if necessary repeatedly) running the MIP program to solve the slicing model.

All the three Python scripts can be run directly from command line by specifying the dimension as first command line argument. E.g. ```python slicing_model.py 4```.
