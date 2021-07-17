#!/usr/bin/env python
# coding: utf-8

# In[52]:

import gc
import sys

import numpy as np
import itertools as it
import math

import gurobipy as grb
from gurobipy import GRB


def generate_testset(d, *, perm=True, flip=True, conn=True, cross_validate=True, check_nedges=True):
    """
    generator that returns containers of hypercube nodes. Nodes are given as tuples of their coordinates.
    e.g.
    return {(0,0,0), (0,0,1)}
    """

    # -- constants --
    def e(i):
        ret = [0] * d
        ret[i - 1] = 1
        return tuple(ret)

    zero = (0,) * d

    hypercube_nodes = set(it.product([0, 1], repeat=d))
    hypercube_edges = [set((p1, p2)) for p1, p2 in it.combinations(hypercube_nodes, r=2)
                       if sum(h1 != h2 for h1, h2 in zip(p1, p2)) == 1]
    max_slicable = (d - d // 2) * math.comb(d, d // 2)

    # -- functions --
    def bitflip(v, pos=None, mask=None):
        """
        pos: from 1 to d
        mask: as list of 0 or 1s, or as string thereof
        return: tuple with some bits flipped
        """
        if pos:
            ret = list(v)
            ret[pos - 1] = ret[pos - 1] ^ 1
            return tuple(ret)
        if mask:
            if type(mask) == str:
                mask = [int(b) for b in mask]
            ret = []
            for i, b in enumerate(mask):
                ret.append(v[i] ^ b)
            return tuple(ret)

    def not_connected(V):
        if len(V) == 0: return False
        V = set(V)
        Q = [V.pop()]
        while len(Q) > 0:
            v = Q.pop()
            for n in nghbrs(v):
                if n in V:
                    V.remove(n)
                    Q.append(n)
        return len(V) != 0

    def nghbrs(X):
        if type(next(iter(X))) != tuple:
            X = [X]
        R = set()
        for v in X:
            for bfi in range(1, d + 1):
                h = bitflip(v, pos=bfi)
                if h not in X:
                    R.add(h)
        return R

    def permute_coords(w):
        """
        w: set of vertices
        return: generator which yields sets which are parallel permutations of the coordinates of the vertices
        """
        inds = list(range(d))
        for p in it.permutations(inds):
            yield {tuple(v[i] for i in p) for v in w}

    def has_cross_violation(tset, X):
        for (w1, w2), (x1, x2) in it.product(it.combinations(tset, 2), it.combinations(X, 2)):
            if all(w1[i] + w2[i] == x1[i] + x2[i] for i in range(d)):
                return True
        return False

    # -- code --

    W = {i: [] for i in range(1, 2 ** (max(d, 3) - 1) + 1)}
    W[1].append(frozenset({zero}))
    W[2].append(frozenset({zero, e(d)}))
    W[3].append(frozenset({zero, e(d), e(d - 1)}))
    if d < 3:
        yield from it.chain(*[W[i] for i in range(1, d + 1)])
    else:
        yield from it.chain(W[1], W[2], W[3])

    for k in range(4, 2 ** (d - 1) + 1):
        H = set()

        for w in W[k - 1]:
            # w is a set (of tuples)
            for v in nghbrs(w):
                a = w.union([v])
                H.add(frozenset(a))
        del W[k - 1]
        gc.collect()

        while len(H) > 0:
            w = H.pop()
            hndiff = hypercube_nodes.difference(w)
            if len(H) > 0:
                flippings = it.product((0, 1), repeat=d) if flip else [(0,) * d]
                for bf in flippings:
                    w_flipped = {bitflip(v, mask=bf) for v in w}
                    permutations = permute_coords(w_flipped) if perm else [w_flipped]
                    for equivalent in permutations:
                        if equivalent in H:
                            H.remove(equivalent)
            W[k].append(w)  # w and all its equivalents were removed from H, add w to list

            nedges = sum(1 for e in hypercube_edges if len(e & w) == 1)
            if nedges <= max_slicable:
                if not not_connected(hndiff):
                    # first check conn, it's worth it (factor 5 faster than check cross)
                    if not has_cross_violation(w, hndiff):
                        # don't yield w if we already know that it doesn't correspond to a 1-slicable set of edges.
                        # however, w has still to be kept in W[k], could later lose cross_violation property.
                        yield w


def get_model(d):
    model = grb.Model()

    eps = model.addVar(vtype=GRB.CONTINUOUS, name="eps")
    a = model.addVars(d, vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0, name="a")
    b = model.addVar(vtype=GRB.CONTINUOUS, lb=-d, ub=d, name="b")

    model.setObjective(eps, GRB.MAXIMIZE)
    model.update()

    return model, eps, a, b


def get_minmax_eps(d, verbosity=2, **kwargs):
    """
    Solve the LP to find max eps for every testset from generate_testset() and return
    smallest such. kwargs are forwarded to generate_testset().
    :param d:
    :param verbosity:
    :param kwargs:
    :return: min max eps
    """
    hypercube_nodes = list(it.product([0, 1], repeat=d))

    minmax_eps = math.inf

    model, eps, a, b = get_model(d)
    model.setParam(GRB.Param.OutputFlag, 0 if verbosity < 3 else 1)

    for tset in generate_testset(d, **kwargs):
        if verbosity >= 3:
            print("New tset: ", tset)

        AM = np.zeros(shape=(2 ** d, d + 2))
        for j, p in enumerate(hypercube_nodes):
            if p in tset:
                AM[j, :] = [1, *p, -1]
            else:
                AM[j, :] = [1, *[-x for x in p], 1]

        for zeile in AM:
            model.addConstr(zeile[0] * eps +
                            grb.quicksum(zeile[i + 1] * a[i] for i in range(d)) +
                            zeile[-1] * b <= 0)

        model.optimize()
        if model.status == GRB.OPTIMAL and model.ObjVal > 0:

            minmax_eps = min(minmax_eps, model.ObjVal)

            avals = [a[i].x for i in range(d)]
            if verbosity >= 2:
                tstr = "{" + ", ".join(["".join(str(i) for i in v) for v in sorted(tset)]) + "}"
                print(f"optimal: {model.ObjVal:.3f}  (for set {tstr})")
                print(f"a: [{', '.join([f'{x:.3f}' for x in avals])}], b: {b.x:.3f}")
            if verbosity >= 1:
                print(f"minmax eps: {minmax_eps:.4f}")
            if verbosity >= 2:
                print()
        else:
            if verbosity >= 2:  print(
                "not feasible. ({" + ", ".join(["".join(str(i) for i in v) for v in sorted(tset)]) + "})\n")

        model.remove(model.getConstrs())

    if verbosity > 0:
        print("\nend result minMax eps:", minmax_eps)

    return minmax_eps


if __name__ == "__main__":
    try:
        d = int(sys.argv[1])
    except:
        print("Provide dimension as command line arg. (e.g. `python bestepsilon_original.py 3`")
    else:
        get_minmax_eps(d, verbosity=2, perm=True, flip=True)
