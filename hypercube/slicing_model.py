
import itertools as it
import math
import sys

import gurobipy as grb
from gurobipy import GRB


class FeasabilityModel:
    def __init__(self, k):
        pass

    def run(self):
        self.model.optimize()

    def admits(self):
        if self.model.status == GRB.LOADED:
            self.run()
        return self.model.status == GRB.OPTIMAL


class SlicingModel(FeasabilityModel):

    def __init__(self, k, d, eps=None, outputFlag=0, logFile=None, refreshRate=20):
        """
        :param k: number of hyperplanes
        :param d: dimension
        :param eps:
        :param outputFlag:
        :param logFile:
        :param refreshRate:
        """

        # actual vertex coordinates as tuples
        points = list(it.product((0,1), repeat=d))
        # tuples of indices of corresponding vertices in points-list
        edges = [(j1,j2) for j1, j2 in it.combinations(range(len(points)), r=2)
                 if sum(h1 != h2 for h1,h2 in zip(points[j1], points[j2])) == 1]
        

        model = grb.Model()
        model.setParam(GRB.Param.OutputFlag, outputFlag)
        model.setParam(GRB.Param.DisplayInterval, refreshRate)
        if logFile:
            model.setParam(GRB.Param.LogFile, logFile)
            model.setParam(GRB.Param.LogToConsole, 0)
            
        M = 2*d
        if not eps:
            if d == 2: eps = 1/2
            if d == 3: eps = 1/4
            if d == 4: eps = 1/6
            if d == 5: eps = 1/10
            if d == 6: eps = 1/14
            if d  > 6: eps = max(1/int(math.sqrt(2+d) * math.sqrt(1+d)**d), model.Params.FeasibilityTol)

        a = model.addVars(k, d, vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0, name="a")
        b = model.addVars(k, vtype=GRB.CONTINUOUS, lb=0.0, ub=d, name="b")
        z = model.addVars(k, len(points), vtype=GRB.BINARY, name="z")
        w = model.addVars(k, len(edges), vtype=GRB.BINARY, name="w")
        
        
        if k >= d:
            # we know a easy solution for this case
            for i in range(k):
                for j in range(d):
                    a[i,j].Start = 1 if i == j else 0
                b[i].Start = 0.5

        model.update()

        # -- Model Constraints --
        for i in range(k):
            for j, p in enumerate(points):
                model.addConstr(grb.quicksum(a[i,index] * p[index] for index in range(d)) - b[i]
                                >= (M + eps) * z[i,j] - M)
                model.addConstr(grb.quicksum(a[i,index] * p[index] for index in range(d)) - b[i]
                                <= (M + eps) * (z[i,j]-1) + M)
        
        for i in range(k):
            for e, edge in enumerate(edges):
                model.addConstr(w[i,e] <= z[i,edge[0]] + z[i,edge[1]])
                model.addConstr(w[i,e] <= 2 - z[i,edge[0]] - z[i,edge[1]])
                model.addConstr(w[i,e] >= z[i,edge[0]] - z[i,edge[1]])
                model.addConstr(w[i,e] >= z[i,edge[1]] - z[i,edge[0]])
                

        for e in range(len(edges)):
            model.addConstr(w.sum('*',e) >= 1)

        # -- wlog fix some variables --
        # get a edge incident to origin and let first hyperplane slice this one
        e = [ei for ei in range(len(edges)) if edges[ei][0]==0][0]
        model.addConstr(w[0,e] == 1)
        model.addConstr(z[0,edges[e][0]] == 0)
        model.addConstr(z[0,edges[e][1]] == 1)
        # origin is always on negative side of planes
        for i in range(k):
            model.addConstr(z[i, 0] == 0)
            
        # -- cutting planes --
        # wlog for each plane at least one point is on its positive side
        for i in range(k):
            model.addConstr(z.sum(i, "*") >= 1)

        model.update()
        
        self.model = model
        self.a = a
        self.b = b
        self.z = z
        self.w = w
        self.d = d
        self.k = k
        self.edges = edges
        self.points = points



if __name__ == "__main__":
    try:
        d = int(sys.argv[1])
    except:
        print("Provide dimension as command line arg. (e.g. `python slicing_model.py 4`")
    else:
        for k in range(math.ceil(5 / 6 * d) - 1, -1, -1):
            print(f"Solving k={k}, d={d} ...")
            if not SlicingModel(k=k, d=d).admits():
                break
            print("Admits!")
        print(f"It takes {k+1} hyperplanes to slice the {d}-hypercube.")


