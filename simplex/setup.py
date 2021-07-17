from typing import List, Set, Dict, Tuple, Iterable, Sequence, Collection, FrozenSet
import time
import itertools as it
import more_itertools as mit
import random
import numpy as np
import math
import os

import gurobipy as grb
from gurobipy import GRB

from utils import vprint

DEBUG = False

DEFAULT_VERBOSITY = 1

PATH_SAVEFILES = os.path.join(os.getcwd(),"savefiles/")
DEFAULT_DATABASE = os.path.join(PATH_SAVEFILES, "database-master.json")
TESTING_FILENAME = "hyperedges-night.obj"

# multiprocessing
N_PROCESSES = 4
