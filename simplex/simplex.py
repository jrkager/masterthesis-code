from setup import *
from hypergraph import *

class ResetModel(grb.Model):
    """
    A gurobi Model class which implements methods to add Extra Constraints and Vars that
    can easily be removed (reset).
    """
    def __init__(self, *args, outputFlag=None, **argv):
        self._extraconstrs = []
        self._extravars = []
        super().__init__(*args, **argv)
        if outputFlag != None:
            self.setParam(GRB.Param.OutputFlag, outputFlag)

    def addExtraConstrs(self, constrs):
        c = self.addConstrs(constrs)
        self._extraconstrs.extend(c.values())

    def addExtraConstr(self, constr):
        c = self.addConstr(constr)
        self._extraconstrs.append(c)

    def resetConstrs(self):
        for c in self._extraconstrs:
            self.remove(c)
        self._extraconstrs = []

    def addExtraVar(self, *args, **argv):
        v = self.addVar(*args, **argv)
        self._extravars.append(v)
        return v

    def addExtraVars(self, *args, **argv):
        v = self.addVars(*args, **argv)
        self._extravars.extend(v.values())
        return v

    def resetVars(self):
        for c in self._extravars:
            self.remove(c)
        self._extravars = []

class ConvResetModel(ResetModel):
    def __init__(self, K, *args, outputFlag=None, **argv):
        """
        create a LP model with K non-negative variables which sum up to 1
        :param K: cardinality of edges to be passed to model (# vertices of edge)
        :return: grb.Model with addition of ResetModel methods and "mus" attribute
        """
        super().__init__(*args, outputFlag=outputFlag, **argv)
        self._mus = self.addVars(K, vtype=GRB.CONTINUOUS, name='mus')
        self.update()
        # by convention continuous vars are non-negative unless otherwise stated
        # model.addConstrs((mus[i] >= 0 for i in range(K)))
        self.addConstr(self._mus.sum() == 1)

    @property
    def mus(self):
        return self._mus


class Simplex:
    @staticmethod
    def getVertices(d):
        ret = np.zeros((d + 1, d), dtype=int)
        ret[1:] = np.identity(d, dtype=int)
        return {tuple(int(vi) for vi in v) for v in ret}

    def __init__(self, d):
        self.d = d
        self.vertices = Simplex.getVertices(d)
        self.models = dict()

    def __str__(self):
        ret = f"Simplex(d={self.d}, models={{{f', '.join([f'{k}: {id(v)}' for k,v in self.models.items()])}}})"
        return ret

    def createModel(self, K):
        self.models[K] = ConvResetModel(K, outputFlag=0)

    def presolve(self, hyperedge):
        """
        Apply easy, certifying checks for non-validity.
        :param hyperedge:
        :return:
        True if hyperedge could(!) still be valid.
        False if hyperedge is(!) not valid
        """
        def is_outside_coordinates(e):
            for j in range(self.d):
                if all(v[j] < 0 for v in e):
                    return True
                if all(v[j] > 1 for v in e):
                    return True
            if all(sum(v) > 1 for v in e):
                return True
            if all(sum(v) < 0 for v in e):
                return True
            return False

        def couldBeHyperedgeLA(edge):  # call only for len(edge) == d!
            # if len(edge) != d: return True
            A = np.array(edge)
            if round(np.linalg.det(
                  A)) == 0:  # origin is an affine combi of the vertices. Round can be used because det is integer
                return True
            res = np.linalg.solve(A, [1] * self.d)
            if all(res < 0.9999):
                return False
            return True

        card = len(hyperedge)

        if card == 2:
            return self.isHyperedgeCard2(hyperedge)

        if card == self.d:
            if is_outside_coordinates(hyperedge): return False
            if not couldBeHyperedgeLA(hyperedge): return False
        else:
            if is_outside_coordinates(hyperedge): return False
        return True

    def isHyperedge(self, hyperedge:Collection):
        """
        Efficiently determine if hyperedge is a valid hyperedge. First some easy presolving
        is done. For card 2 a direct method is used. Otherwise gurobi is started
        :param hyperedge:
        :return: boolean
        """
        hyperedge = tuple(hyperedge)
        if not self.presolve(hyperedge):
            if DEBUG:
                assert not self.isHyperedgeGurobi(hyperedge)
            return False
        elif len(hyperedge) == 2:
            if DEBUG:
                assert self.isHyperedgeGurobi(hyperedge)
            return True
        return self.isHyperedgeGurobi(hyperedge)

    def isHyperedgeCard2(self, hyperedge):
        """
        checks if a hyperedge (or: edge) with cardinality 2 (i.e. it spans between two vertices)
        cuts the standard simplex in dimension d
        :param hyperedge: tuple or list of vertices
        :param d: dimension
        :return: boolean
        """
        e = hyperedge
        for i in range(self.d):
            if e[0][i] * e[1][i] < 0:
                if e[0][i] < e[1][i]:
                    nenner, zaehler = -(e[0][i] - e[1][i]), -e[0][i]
                else:
                    nenner, zaehler = e[0][i] - e[1][i], e[0][i]
                vals = [nenner * e[0][j] + zaehler * (e[1][j] - e[0][j]) for j in range(self.d) if j != i]
                if all(val >= 0 for val in vals) and \
                      sum(vals) <= nenner:
                    return True
        return False

    def isHyperedgeGurobi(self, hyperedge):  # reuse model possibly 35% faster
        """
        lazy instantiate a model (on first call) and reuse for further calls.
        model is left untouched after execution

        If you use parallel processing make sure to create the simplex object as a global in the initialiazer
        of the pool-workers. Creating the model with simplex.createModel(K) before batch-calling this method
        doesn't work since gurobi-models cannot be pickled.

        :param hyperedge:
        :return: boolean
        """
        K = len(hyperedge)
        if K not in self.models:
            # possibly 35% faster if reusing model and vars
            self.createModel(K)
            vprint(4, "DEBUG Simplex.isHyperedge(): creating a model for card", K, id(self.models[K]))
        model = self.models[K]
        mus = model.mus

        model.addExtraConstrs(grb.quicksum(mus[i] * hyperedge[i][j]
                                  for i in range(K))
                              >= 0 for j in range(self.d))
        model.addExtraConstr(grb.quicksum(mus[i] * hyperedge[i][j]
                                 for i in range(K)
                                 for j in range(self.d))
                             <= 1)

        model.optimize()
        ret = model.status == GRB.OPTIMAL

        model.resetConstrs()

        return ret

    def findHyperedgeMinCoeffMaxDeg(self, vertices: Collection[Tuple[int,...]], vertDegs: Dict):
        """
        using a LP minimizing number of non-zero coefficients, find a valid (simplex-crossing) hyperedge in vertices
        which uses the fewest possible number of vertices (or: with the smallest possible cardinality).
        If a simplex-vertex is in vertices then the returned edge will be a tuple consisting of only this
        vertex. Watch out to remove those vertices before!
        :param vertices: sequence of vertices
        :return: tuple of vertices that span the found hyperedge or None if the convex hull of the vertices doesn't
        intersect the simplex.
        """
        #
        vertices = list(vertices)
        K = len(vertices)
        if K not in self.models:
            self.createModel(K)
        model = self.models[K]
        mus = model.mus

        x = model.addExtraVars(K, vtype=GRB.BINARY, name='x')

        model.update()
        model.addExtraConstrs((grb.quicksum(mus[i] * vertices[i][j]
                                            for i in range(K))
                               >= 0 for j in range(self.d)))
        model.addExtraConstr(grb.quicksum(mus[i] * vertices[i][j]
                                          for i in range(K)
                                          for j in range(self.d))
                             <= 1)
        model.addExtraConstrs((x[i] >= mus[i]
                               for i in range(K)))

        model.ModelSense = GRB.MINIMIZE

        model.setObjectiveN(x.sum(), index=0, priority=1) # higher priority

        model.setObjectiveN(sum(x[i] * vertDegs[vertices[i]]
                                for i in range(K)),
                            index=1, priority=0, weight=-1) # lower priority

        model.optimize()

        if model.status != GRB.OPTIMAL:
            ret = None
        else:
            ret = frozenset(it.compress(vertices, [round(v.x) for v in x.values()]))
        model.resetConstrs()
        model.resetVars()

        return ret

    def findHyperedgeMinCoeffMaxMinDeg(self, vertices: Collection[Tuple[int,...]], vertDegs: Dict):
        vertices = list(vertices)
        K = len(vertices)
        if K not in self.models:
            self.createModel(K)
        model = self.models[K]
        mus = model.mus

        x = model.addExtraVars(K, vtype=GRB.BINARY, name='x')
        z = model.addExtraVar(name='z')

        model.update()
        model.addExtraConstrs((grb.quicksum(mus[i] * vertices[i][j]
                                            for i in range(K))
                               >= 0 for j in range(self.d)))
        model.addExtraConstr(grb.quicksum(mus[i] * vertices[i][j]
                                          for i in range(K)
                                          for j in range(self.d))
                             <= 1)
        model.addExtraConstrs((x[i] >= mus[i]
                               for i in range(K)))

        M = max(vertDegs.values())
        model.addExtraConstrs((z <= vertDegs[vertices[i]] + (1-x[i]) * M
                                   for i in range(K)))

        model.ModelSense = GRB.MINIMIZE

        model.setObjectiveN(x.sum(), index=0, priority=1) # higher priority

        model.setObjectiveN(z, #maximize
                            index=1, priority=0, weight=-1) # lower priority

        model.optimize()

        # print([v.x for v in x.values()])
        # print([v.x for v in mus.values()])
        if model.status != GRB.OPTIMAL:
            ret = None
        else:
            ret = frozenset(it.compress(vertices, [round(v.x) for v in x.values()]))
        model.resetConstrs()
        model.resetVars()

        return ret

    def findHyperedgeMinCoeff(self, vertices: Collection[Tuple[int,...]]):
        """
        using a LP minimizing number of non-zero coefficients, find a valid (simplex-crossing) hyperedge in vertices
        which uses the fewest possible number of vertices (or: with the smallest possible cardinality).
        If a simplex-vertex is in vertices then the returned edge will be a tuple consisting of only this
        vertex. Watch out to remove those vertices before!
        :param vertices: sequence of vertices
        :return: tuple of vertices that span the found hyperedge or None if the convex hull of the vertices doesn't
        intersect the simplex.
        """
        #
        vertices = list(vertices)
        K = len(vertices)
        model = ConvResetModel(K, outputFlag=0)
        mus = model.mus

        x = model.addExtraVars(K, vtype=GRB.BINARY, name='x')
        model.update()
        model.addExtraConstrs((grb.quicksum(mus[i] * vertices[i][j]
                                   for i in range(K))
                               >= 0 for j in range(self.d)))
        model.addExtraConstr(grb.quicksum(mus[i] * vertices[i][j]
                                 for i in range(K)
                                 for j in range(self.d))
                             <= 1)
        model.addExtraConstrs((x[i] >= mus[i]
                               for i in range(K)))

        model.setObjective(x.sum(), GRB.MINIMIZE)
        model.optimize()

        # print([v.x for v in x.values()])
        # print([v.x for v in mus.values()])
        if model.status != GRB.OPTIMAL:
            ret = None
        else:
            ret = frozenset(it.compress(vertices, [round(v.x) for v in x.values()]))
        model.resetConstrs()
        model.resetVars()

        return ret


    def isCertifiedBy(self, hypergraph):
        """
        Is the passed hypergraph certifying for the dimension of this simplex?
        (i.e., is its chromatic number >= d+1)
        :param hypergraph:
        :return:
        """
        if self.d != hypergraph.d:
            raise TypeError("Simplex and Hypergraph are not in the same dimension.")
        return hypergraph.isCertifying()