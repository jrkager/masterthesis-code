import pylgl
import networkx as nx
from collections import Counter

from utils import vprint, timeit, savefig, figsize
import matplotlib as mpl

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"
    }
mpl.rcParams.update(pgf_with_latex)


# mpl.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

import warnings



from setup import *
import simplex

# new type
HyperedgeType = FrozenSet[Tuple[int,...]]

class DimensionException(Exception):
    pass

class Plotgraph:

    def __init__(self, hgraph=None, color_dict = None,
                highlight_edges = None,
                title=None, iter=None, depthshade=None):
        self.hgraph = hgraph
        self.color_dict = color_dict
        self.highlight_edges = highlight_edges
        self.title = title
        self.iter = iter
        self.depthshade = depthshade

class Plotpool:

    queue = []

    @staticmethod
    def plot3D():
        # fig, axs = plt.subplots(math.ceil(len(Plotpool.queue) / 2), 2, subplot_kw={'projection': '3d'})
        fig = plt.figure(figsize=figsize(scale=1.1, scale_height=0.4+0.6*(len(Plotpool.queue)+1)//2))
        rows = math.ceil(len(Plotpool.queue) / 2)
        cols = 2
        for i, plgraph in enumerate(Plotpool.queue):
            plgraph = Plotpool.queue[i]
            plgraph.hgraph.plot_3d(color_dict=plgraph.color_dict,
                            highlight_edges=plgraph.highlight_edges,
                            title=plgraph.title,
                            fig=fig,
                            ax=fig.add_subplot(rows, cols, i+1, projection="3d"),
                            show_plot=False,
                            depthshade=plgraph.depthshade)
        fig.tight_layout()
        savefig(plt, "-".join(str(p.iter) for p in Plotpool.queue))

    @staticmethod
    def reset():
        Plotpool.queue = []

class Hypergraph:
    def __init__(self, d=None, hyperedges=set(), vertices=None): # hyperedges: Collection[FrozenSet[Tuple[int, ...], ...], ...]
        if not any((d, hyperedges)):
            raise Exception("Provide at least one of dim or hyperedges")
        self.d = d or len(next(iter(next(iter(hyperedges)))))
        self._hyperedges = set(frozenset(e) for e in hyperedges)
        self._vertices = collect_vertices(hyperedges)
        if vertices:
            self._vertices.update(vertices)
        self._certifying = None

    def __bool__(self):
        return True #otherwise a hypergraph object with >=1 vertices and 0 edges will evaluate to false

    def __len__(self):
        return len(self.hyperedges)

    def copy(self):
        return Hypergraph(self.d, self._hyperedges.copy(), self._vertices.copy())

    def __eq__(self, other):
        return self.vertices == other.vertices and self.hyperedges == other.hyperedges

    def __sub__(self, other):
        if self.d != other.d:
            raise DimensionException("Can only subtract graphs of equal dimension.")
        return Hypergraph(self.d, self.hyperedges-other.hyperedges)

    def __add__(self, other):
        if self.d != other.d:
            raise DimensionException("Can only add graphs of equal dimension.")
        return Hypergraph(self.d, self.hyperedges|other.hyperedges)

    def __or__(self, other):
        return self + other

    def __le__(self, other):
        if self.d != other.d:
            raise DimensionException("Can only compare graphs of equal dimension.")
        return self.hyperedges <= other.hyperedges

    def __ge__(self, other):
        return other <= self

    def __and__(self, other):
        if self.d != other.d:
            raise DimensionException("Can only cut graphs of equal dimension.")
        return Hypergraph(self.d, self.hyperedges & other.hyperedges)

    @property
    def hyperedges(self):
        return self._hyperedges

    @hyperedges.setter
    def hyperedges(self, he):
        self._certifying = None
        if he is None:
            self._hyperedges = set()
        else:
            self._hyperedges = set(frozenset(e) for e in he)
            self._vertices.update(*he)

    def addEdge(self, edge:Sequence[Tuple[int,...]]): # edge: FrozenSet[Tuple[int, ...], ...]
        self._certifying = None
        self.hyperedges.add(frozenset(edge))
        self._vertices.update(edge)

    def addEdges(self, edges: Collection[Sequence[Tuple[int,...]]]):
        self._certifying = None
        self.hyperedges.update(frozenset(e) for e in edges)
        self._vertices.update(*edges)

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        """
        Sets vertices to given iterable. Vertices from existent hyperedges are always added in addition.
        If vertices is None, vertices will be reset to the ones found in hyperedges.
        :param vertices:
        :return:
        """
        if not vertices:
            self.resetVertices()
            return
        self._vertices = set(vertices)
        self._vertices.update(*self._hyperedges)

    def addVertices(self, vertices):
        self._vertices.update(vertices)

    def resetVertices(self):
        """
        reduce vertices-set to the ones that occur in the hyperedges
        :return:
        """
        self._vertices = collect_vertices(self._hyperedges)

    def isCertifying(self):
        if self._certifying is not None:
            return self._certifying
        self._certifying = not self.admitsKColoring(self.d)
        return self._certifying

    def admitsKColoring(self, k):
        """
        Using a SAT with lingeling, determine if the hypergraph is k colorable.
        :param k:
        :return: boolean
        """
        return admitsKColoring(self._hyperedges, k, self._vertices)

    def getKColoring(self, k):
        """
        return a dictionary whose keys are numbers from 1 to k and the values are
        lists of vertices (lists of tuples). If Graph is not k colorable, return None.
        :param k:
        :return:
        """
        return admitsKColoring(self._hyperedges, k, self._vertices, return_coloring=True) or None

    def calcChromaticNumber(self, start_at=-1):
        """
        start_at > 0: check if graph admits coloring for k counting down from start_at to 1. Stop when graph
                        doesn't admit coloring anymore
        start_at == -1: count up from 1 to a max k (currently 25) (or until graph starts to admit coloring).
                        If chromatic number is more than max k it will be signalized
        return: -1 if chromatic number could not be determined (start_at too low or upper bound reached)
                        else: chromatic number

        example: check if hg is certifying (doesn't admit d coloring):
                chr_n = hg.calcChromaticNumber(start_at=args.d)
                if chr_n == -1: # this means that it was > args.d
                    chr_n = args.d + 1
        """
        ub = 20

        if not self._hyperedges:
            if not self._vertices: return 0
            else: return 1
        if start_at == -1:
            k = 1
            while k <= ub and not self.admitsKColoring(k):
                vprint(2, "Graph doesn't admit {} coloring.".format(k))
                k += 1
            if k > ub:
                vprint(2, "Upper bound reached, chromatic number is bigger than", k - 1)
                return -1
            else:
                vprint(2, "Graph admits {} coloring.".format(k))
                return k - 1
        else:
            k = start_at
            starttime = time.time()
            while k >= 1 and self.admitsKColoring(k):
                vprint(2, "Graph admits {} coloring. ({:.2f}s)".format(k, time.time() - starttime))
                k -= 1
                starttime = time.time()
            vprint(2, "Graph doesn't admit {} coloring. ({:.2f}s)".format(k, time.time() - starttime))
            return k + 1 if k < start_at else -1

    def lowerBoundChromaticNumber(self):
        """
        (faerbungszahl LP)
        Using a LP to find a quick lower bound to the chromatic number
        :return: chromatic number lower bound as integer
        """
        dlp_model = grb.Model()
        dlp_model.setParam(GRB.Param.OutputFlag, 0.2)
        x = dlp_model.addVars(self.vertices, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='x')
        dlp_model.update()

        ip_model = grb.Model()
        z = ip_model.addVars(self.vertices, vtype=GRB.BINARY, name='z')
        ip_model.update()
        ip_model.addConstrs((grb.quicksum(z[v] for v in e) <= len(e) - 1
                             for e in self.hyperedges))

        add_stable_set = None  # given as set of vertices (tuples)
        beta = math.inf
        while beta > 1:
            if add_stable_set:
                vprint(3, "Add stable set:", add_stable_set)
                dlp_model.addConstr(grb.quicksum(x[vert] for vert in add_stable_set) <= 1)
            dlp_model.setObjective(x.sum(), GRB.MAXIMIZE)
            dlp_model.optimize()
            if dlp_model.status != GRB.OPTIMAL:
                vprint(1, "DLP not solvable")
                return -1
            vprint(3, "x = ", [v.x for v in x.values()])

            ip_model.setObjective(grb.quicksum(z[v] * x[v].x for v in self.vertices), GRB.MAXIMIZE)
            ip_model.optimize()
            add_stable_set = [k for k, var in z.items() if round(var.x) == 1]
            beta = ip_model.objVal
            vprint(3, "Beta:", beta)

        tolerance = 1e-3
        vprint(2, f"Added {len(dlp_model.getConstrs())} stable set constraints")
        vprint(1, "chrom. number >= alpha =",
               f"{dlp_model.objVal:.4f}",
               "🎉 " if math.ceil(dlp_model.objVal - tolerance) == self.d + 1 else "❌ ")

        return math.ceil(dlp_model.objVal - tolerance)

    def weakLowerBoundChromaticNumber(self):
        """
        too weak in most cases. Use lowerBoundChromaticNumber()
        :return: lower bound (integer)
        """
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        for he in self.hyperedges:
            if len(he) > 2:
                vprint(1, "Can't run biggest stable set with hypergraphs")
                return -1
            G.add_edge(*he)
        GC = nx.algorithms.operators.complement(G)
        # size of largest clique in complement equals size of largest stable set in original graph
        cl_number = nx.algorithms.clique.graph_clique_number(GC)
        print("clique number (size of largest clique)", cl_number)
        guess = math.ceil(len(self.vertices) / cl_number)
        print("chrom. number >= |V|/alpha(G) =",
              guess,
              "🎉 " if guess == self.d + 1 else "❌ ")
        return guess


    def collect_vertices(self):
        return {v for e in self._hyperedges for v in e}

    def getHyperedgesStats(self):
        if len(self._hyperedges) == 0:
            return dict()
        edgelengths = [len(e) for e in self._hyperedges]
        return {i: edgelengths.count(i)
                    for i in range(2, max(edgelengths) + 1)}

    def getVerticesStats(self):
        if len(self._vertices) == 0:
            return dict()
        deg_vert = sorted([sum(1 for e in self._hyperedges if v in e) for v in self._vertices])
        len_deg_vertex = Counter(deg_vert)
        return dict(len_deg_vertex)

    def print_stats(self, v=None):
        if not self._hyperedges and not self._vertices:
            vprint(v,"Empty hypergraph")
            return
        if self._hyperedges:
            vprint(v,f"Anzahl Hyperkanten: {len(self._hyperedges)} ({self.getHyperedgesStats()})")
        if self._vertices:
            vertStats = self.getVerticesStats()
            vsmin=min(vertStats)
            vsmax=max(vertStats)
            vprint(v,f"Anzahl Knoten: {len(self._vertices)} ({len(self.collect_vertices())}) "
                     f"({vsmin}:{vertStats[vsmin]},...,{vsmax}:{vertStats[vsmax]})")
        vprint(3,f"edges:\n{self._hyperedges}")
        vprint(3,f"vertices:\n{self._vertices}")


    def _scatter_vertices(self, ax, color_dict=None, depthshade=False,markersize=36,**options):
        if not color_dict:
            color_dict = {1 : self._vertices}
        if type(color_dict) is not dict:
            color_dict = {1 : color_dict}

        highlight_verts = color_dict.get("new", [])
        collapse_color_dict = {}
        i = 0
        for v in color_dict.values():
            if v:
                collapse_color_dict[i] = v
                i += 1

        # norm = mpl.colors.Normalize(vmin=0, vmax=len(collapse_color_dict)-1)
        norm = lambda c: c
        # cmap = mpl.cm.get_cmap("Spectral")
        cmap = mpl.cm.get_cmap("tab10")
        markeroptions = {"marker":"o", "edgecolor":"none",
                         "s":markersize
                         }
        markeroptions.update(options)
        if self.d == 3:
            markeroptions["depthshade"] = depthshade
        for c, vertices in collapse_color_dict.items():
            ax.scatter(*zip(*vertices), color=cmap(norm(c)), **markeroptions)
        if highlight_verts:
            ax.scatter(*zip(*highlight_verts), color="lime", **markeroptions)

        return set().union(*color_dict.values())

    def plot_2d(self, color_dict = None,
                highlight_edges = None,
                title=None):
        """

        :param color_dict: keys for different colors (any hashable object)
        and values for collections of vertices
        :param highlight_edges:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)



        # plot (hyper)edges
        if not highlight_edges:
            highlight_edges = set()
        patches = []
        patches_hl = []
        for e in self._hyperedges:
            # plot plain edges (card 2)
            if len(e) == 2:
                if e in highlight_edges:
                    ax.plot(*zip(*e), c="lime")
                else:
                    ax.plot(*zip(*e), c="gray", linewidth=0.95)
            else:
                if e in highlight_edges:
                    patches.append(Polygon(e, True))
                else:
                    patches_hl.append(Polygon(e, True))

        p = PatchCollection(patches, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(patches_hl, alpha=0.4)
        p.set_color("lime")
        ax.add_collection(p)

        simplex_vertices = list(simplex.Simplex.getVertices(self.d))
        ax.add_collection(PatchCollection([Polygon(simplex_vertices, True)], alpha=0.5))

        # plot vertices with color
        vertices = self._scatter_vertices(ax, color_dict, zorder=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect(1)

        axmin = [min(vs) for vs in zip(*vertices)]
        axmax = [max(vs) for vs in zip(*vertices)]
        pad = 0.15
        ax.axes.set_xlim(  left=axmin[0]-pad, right=axmax[0]+pad)
        ax.axes.set_ylim(bottom=axmin[1]-pad,   top=axmax[1]+pad)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.set_xticks(range(axmin[0], axmax[0]+1))
        ax.set_yticks(range(axmin[1], axmax[1]+1))
        ax.grid(True, which='major', linestyle="--", linewidth=0.6)

        ax.set_axisbelow(True)

        if title: fig.canvas.set_window_title(title)

        def press(event):
            print('press', event.key)
            if event.key == "f":
                fig.tight_layout()
                savefig(plt, f"{self.d},{len(self._vertices)},{len(self._hyperedges)}")
            elif event.key == "q":
                plt.close()

        fig.canvas.mpl_connect('key_press_event', press)

        plt.show()

    def plot_3d(self, color_dict = None,
                highlight_edges = None,
                title=None, fig=None, ax=None, show_plot=True, iter=None, depthshade=None):

        if not fig:
            fig = plt.figure()
            ax = None

        # ax = Axes3D(fig)
        if not ax:
            ax = fig.add_subplot(111, projection='3d')

        if depthshade is None and not hasattr(self, "depthshade"):
            self.depthshade = True
        if depthshade is None:
            depthshade = self.depthshade
        # plot vertices with color
        vertices = self._scatter_vertices(ax, color_dict, depthshade=depthshade, markersize=40)

        # plot (hyper)edges
        if not highlight_edges:
            highlight_edges = set()
        for e in self._hyperedges:
            # plot plain edges (card 2)
            if len(e) == 2:
                if e in highlight_edges:
                    ax.plot(*zip(*e), c="lime", linewidth=1.1)
                else:
                    ax.plot(*zip(*e), c="gray", linewidth=0.5)

                # plot inner
                a, b = e
                inters = []
                for i in range(self.d):
                    if a[i] == b[i]:
                        continue
                    x = []
                    for j in range(self.d):
                        x.append(a[j] + a[i] * (b[j] - a[j]) / (a[i] - b[i]))
                    if sum(x) <= 1 and all(xi >= 0 for xi in x):
                        inters.append(tuple(x))
                if sum(a) - sum(b) != 0:
                    alpha = (1 - sum(b)) / (sum(a) - sum(b))
                    x = [alpha * a[j] + (1 - alpha) * b[j] for j in range(self.d)]
                    if all(xi >= 0 for xi in x):
                        inters.append(tuple(x))

                if not all(xi in (0,1) for xi in x):
                    # self._scatter_vertices(ax, inters)
                    ax.plot(*zip(*inters), c="red", alpha=0.6, linewidth=0.6)
            else:
                # print("hiding hyperedge in 3d for now")
                ax.add_collection3d(Poly3DCollection(e, alpha =0.03, color="gray"))

        simplex_vertices = simplex.Simplex.getVertices(self.d)
        for vset in it.combinations(simplex_vertices, 3):
            ax.add_collection3d(Poly3DCollection(vset, alpha = 0.15, color="blue")) #color="blue"
        for vset in it.combinations(simplex_vertices, 2):
            ax.plot(*zip(*vset), c="blue", alpha=0.5, linewidth=0.9) # c="blue"

        axmin = [min(vs) for vs in zip(*vertices)]
        axmax = [max(vs) for vs in zip(*vertices)]
        pad = 0.15
        ax.axes.set_xlim3d(  left=axmin[0]-pad, right=axmax[0]+pad)
        ax.axes.set_ylim3d(bottom=axmin[1]-pad,   top=axmax[1]+pad)
        ax.axes.set_zlim3d(bottom=axmin[2]-pad,   top=axmax[2]+pad)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xticks(range(axmin[0], axmax[0]+1))
        ax.set_yticks(range(axmin[1], axmax[1]+1))
        ax.set_zticks(range(axmin[2], axmax[2]+1))
        ax.yaxis.grid(True, which='major')

        padding=-3
        ax.xaxis.set_tick_params(pad=padding)
        # ax.tick_params(axis='x', which='major', pad=-3)
        ax.yaxis.set_tick_params(pad=padding)
        ax.zaxis.set_tick_params(pad=padding)

        if "Iteration" in title:
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_zlabel(None)

        if title:
            fig.canvas.set_window_title(title)
            if "Iteration" in title:
                ax.set_title(title, y=0)


        # ax.view_init(azim=16, elev=28)
        # ax.view_init(azim=13, elev=6)
        ax.view_init(azim=14, elev=12)

        self.save_graph = False
        def press(event):
            print('press', event.key)
            if event.key == "f":
                fig.tight_layout()
                savefig(plt, f"{len(self._vertices)},{len(self._hyperedges)}")
            elif event.key == "a":
                Plotpool.queue.append(Plotgraph(self.copy(), color_dict=color_dict,
                                                highlight_edges=highlight_edges,
                                                title=title, iter=iter, depthshade=self.depthshade))
            elif event.key == "p":
                Plotpool.plot3D()
            elif event.key == "d":
                self.depthshade = False
                plt.close()
                self.plot_3d(color_dict, highlight_edges, title=title, iter=iter, fig=fig,ax=ax)
            elif event.key == "D":
                self.depthshade = True
                plt.close()
                self.plot_3d(color_dict, highlight_edges, iter=iter,title=title,fig=fig,ax=ax)
            elif event.key == "q":
                plt.close()
            elif event.key == "y":
                self.save_graph = True

        fig.canvas.mpl_connect('key_press_event', press)

        if show_plot:
            plt.show()

        if self.save_graph:
            return True
        return False


    def plot(self, **kwargs):
        with warnings.catch_warnings():
            if self.d == 2:
                return self.plot_2d(**kwargs)
            if self.d == 3:
                return self.plot_3d(**kwargs)
        return False


def admitsKColoring(hyperedges, k, vertices=None, return_coloring=False): #using SAT with lingeling
    # assert hyperedges and len(hyperedges) and isinstance(hyperedges.pop().pop(), tuple) #vertices must be tuples
    if any(len(e) == 1 for e in hyperedges):
        print("Watch out! There are hyperedges of card 1 in your graph!")

    if not vertices:
        vertices = collect_vertices(hyperedges)
    vertices = list(vertices)


    v_i_dict = {v : i+1 for i,v in enumerate(vertices)}
    n = len(vertices) # n vertices used for this hypergraph
    # vars from 1 to n*k
    var = lambda i, j: (i-1) * k + j
    ij = lambda var: (1 + ((var-1) // k), 1 + ((var-1) % k))

    clauses = []
    # n_vars = n * k # x_1^1, \overline{x_1^2}, ..., x_n^(k-1), x_n^k == 1, -2, 3, 4, ..., n_vars

    # at least one color per v:
    for i in range(1, n + 1):
        clauses.append([var(i, j) for j in range(1, k + 1)])
    # max. one color per v
    for i in range(1, n + 1):
        for j in range(1, k):
            for l in range(j + 1, k + 1):
                clauses.append([-var(i, j), -var(i, l)])
    # at least 2 colors per hyperedge
    for l in range(1, k + 1):
        for p in hyperedges:
            clauses.append([var(v_i_dict[v], j) for v in p for j in range(1, k + 1) if j != l])
    vprint(3, "Clauses prepared (k=%d)." % k)

    result = pylgl.solve(clauses)
    if result == 'UNSAT' or result == 'UNKNOWN':
        return False
    if not return_coloring:
        return True
    ret = {j : [] for j in range(1,k+1)} # {color-id : vertices, ...}
    for v in result:
        if v < 0: continue
        i, j = ij(v)
        ret[j].append(vertices[i-1])
    return ret


def collect_vertices(hedges):
    return {v for e in hedges for v in e}


def issup(p: Collection, sedges:Set[HyperedgeType]):
    for c in range(2, len(p)):
        for sub in it.combinations(p, c):
            if frozenset(sub) in sedges:
                return True
    return False


def get_without_supersets_of_valid_gen(edges: Iterable[HyperedgeType], valid: Set[HyperedgeType]):
    for p in edges:
        if len(p) <= 2:
            yield p
        elif not issup(p, valid):
            yield p


def get_without_supersets_of_valid(edges: Iterable[HyperedgeType],
                                   valid: Set[HyperedgeType]):
    return list(get_without_supersets_of_valid_gen(edges, valid))


def get_without_supersets(edges: Iterable[HyperedgeType]):
    sedges = set(edges)
    return get_without_supersets_of_valid(sedges, sedges)
