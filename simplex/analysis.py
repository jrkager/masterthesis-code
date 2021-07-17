# coding: utf-8

# my scripts
import hypergraph
import utils
from hypergraph import admitsKColoring, collect_vertices, get_without_supersets
from setup import *
from utils import timeit


@timeit
def biggestStableSet(hg: hypergraph.Hypergraph):
    return hg.weakLowerBoundChromaticNumber()

@timeit
def chromaticNumberLP(hg: hypergraph.Hypergraph):
    return hg.lowerBoundChromaticNumber()

def stable_set_analysis(hgraph: hypergraph.Hypergraph):
    """

    :param hgraph: a certifying hgraph
    :return:
    """
    hg = hgraph.copy()
    hg.resetVertices()

    print("\nFinde groesstes Stable set und betrachte ratio vertices/stable sets:")
    biggestStableSet(hg)

    print("Faerbungszahl-LP (neue methode):")
    if chromaticNumberLP(hg) >= hg.d+1:
        vprint(0, "LP successful")
    else:
        vprint(0, "LP not successful")

def stable_set_analysis_weak(hgraph: hypergraph.Hypergraph):
    hg = hgraph.copy()
    hg.resetVertices()

    print("\nFinde groesstes Stable set und betrachte ratio vertices/stable sets:")
    if biggestStableSet(hg) >= hg.d+1:
        vprint(0, "LP successful")
    else:
        vprint(0, "LP not successful")


class Ausduennen:
    @staticmethod
    @timeit
    def randomEdges(hgraph):
        he = hgraph.hyperedges
        lenhe = len(he)
        # start_time = time.time()
        h_shuffled = random.sample(he, lenhe)
        i = 0
        while i < len(h_shuffled):
            utils.print_progress(i, len(h_shuffled))
            if not admitsKColoring(h_shuffled[:i] + h_shuffled[(i + 1):], hgraph.d):
                # it works also without edge i
                h_shuffled.pop(i)
            else:
                i += 1
        utils.print_progress(1, 1)
        he = h_shuffled
        print("\ngeloescht: {}".format(lenhe - len(h_shuffled)))
        # print("\ngeloescht: {} (in {:.2f}s)".format(lenhe - len(h_shuffled), time.time() - start_time))
        hgraph.hyperedges = he
        hgraph.resetVertices()

    @staticmethod
    def randomVertices(hgraph):
        he = hgraph.hyperedges
        lenhe = len(he)
        lenve = len(hgraph.vertices)
        start_time = time.time()
        order = random.sample(hgraph.vertices, lenve)
        for i, shuff_vert in enumerate(order):
            utils.print_progress(i, lenve)
            temp = [e for e in he if not shuff_vert in e]
            if not admitsKColoring(temp, hgraph.d):
                he = temp
        print("\ngeloescht: {} (in {:.2f}s)".format(lenhe - len(he), time.time() - start_time))
        hgraph.hyperedges = he
        hgraph.resetVertices()

    @staticmethod
    def vertices(hgraph, rev=False):
        he = hgraph.hyperedges
        lenhe = len(he)
        start_time = time.time()
        print("kicking card ", end='')
        currcard = -1
        while True:
            verts = collect_vertices(he)
            s = sorted([(v, sum(1 for e in he if v in e)) for v in verts], key=lambda v: v[1], reverse=rev)
            for si in s:
                vert = si[0]
                temp = [e for e in he if not vert in e]
                if not admitsKColoring(temp, hgraph.d):
                    if si[1] != currcard:
                        print(si[1], " ", end='')
                        currcard = si[1]
                    he = temp
                    break
            else:
                # for loop didn't break
                break
        print("\ngeloescht: {} (in {:.2f}s)".format(lenhe - len(he), time.time() - start_time))
        hgraph.hyperedges = he
        hgraph.resetVertices()

    @staticmethod
    def verticesAsc(hgraph):
        Ausduennen.vertices(hgraph, rev = False)

    @staticmethod
    def verticesDesc(hgraph):
        Ausduennen.vertices(hgraph, rev = True)

    @staticmethod
    def edges(hgraph, rev=True):
        he = hgraph.hyperedges
        lenhe = len(he)
        start_time = time.time()
        h_sorted = sorted(he, key=lambda e: len(e), reverse=rev)

        i = lenhe//2
        while admitsKColoring(h_sorted[i:], hgraph.d):
            if i == 1:
                i = 0
                break
            i = i // 2
        h_sorted = h_sorted[i:]
        strike, striking = i, i > 0
        i = 0
        while i < len(h_sorted):
            if not admitsKColoring(h_sorted[:i] + h_sorted[(i + 1):], hgraph.d):
                if striking:
                    strike += 1
                # it works also without edge i
                h_sorted.pop(i)
            else:
                striking = False
                i += 1
        he = h_sorted
        print("start strike:", strike)
        print("\ngeloescht: {} (in {:.2f}s)".format(lenhe - len(h_sorted), time.time() - start_time))
        hgraph.hyperedges = he
        hgraph.resetVertices()

    @staticmethod
    def edgesAsc(hgraph):
        Ausduennen.edges(hgraph, rev = False)

    @staticmethod
    def edgesDesc(hgraph):
        Ausduennen.edges(hgraph, rev = True)

    @staticmethod
    def withoutSupersets(hgraph):
        hgraph.hyperedges = get_without_supersets(hgraph.hyperedges)
        hgraph.resetVertices()

    @staticmethod
    def stableSetAnalysis(hgraph):
        stable_set_analysis(hgraph)

    @staticmethod
    def stableSetAnalysisWeak(hgraph):
        stable_set_analysis_weak(hgraph)
