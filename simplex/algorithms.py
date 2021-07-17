
from setup import *
import savefile_io
from hypergraph import issup, Hypergraph
from savefile_io import Savefile
from shapes import createPyramid, pyramidSlice
from utils import vprint, print_progress, vwarn, question


def grow_starter(hgraph, start_ext, max_ext, max_card, n_facets, slice_func, base_shape_func, simpl, shapename, args,
                 savefile):
    """
    Start the hypergraph classic grow - algorithm
    :param hgraph: empty or prefilled hypergraph
    :param start_ext:
    :param max_ext:
    :param max_card:
    :param n_facets: facets of the shape
    :param slice_func: function that creates a slice of the shape and takes two arguments: extents and number of added side
    (e.g. lambda e, a: pyramidSlice(e, added_side=a))
    :param base_shape_func: function that createes a shape of two arguments: extents and dimension
    (e.g. lambda e, d: createPyramid(e, d))
    :param simpl: a object of Simplex class
    :param shapename: string short name of shape
    :param args: command line args
    :param savefile: a savefile object
    :return:
    """

    # add all vertices in a shape with start_extent.
    # Always remove the vertices from the d-dim-simplex, since they don't prove anything (would need d+1 cols but
    #  doesn't prove that the relaxation complexity is d+1).
    # initialize starting shape
    extents = [start_ext] * n_facets
    hgraph.vertices = base_shape_func(start_ext, args.d)
    # hgraph.vertices = set(random.sample(hgraph.vertices,k=len(hgraph.vertices)))
    # sf.additional["shuffled_vertices"] = True
    vprint(1, f"Starter: {shapename} with {len(hgraph.vertices)} vertices (ext {start_ext})")
    # find all valid edges in starting pyramid:
    counter, total = 0, sum(math.comb(len(hgraph.vertices), k) for k in range(2, max_card + 1))
    step = total // 1000
    if step == 0:
        step = 1
    c = 1
    while c < max_card:# and not hgraph.isCertifying():
        c += 1
        # increment cards for whole starting pyramid and add all new combs as hyperedges
        add_edges = []
        for e in it.combinations(hgraph.vertices, c):
            if c == 2 or not issup(e, hgraph.hyperedges):
                if simpl.isHyperedge(e):
                    add_edges.append(e)
            counter += 1
            if counter % step == 0:
                print_progress(counter, total)
        print("init: card {} adds {} valid edges".format(c, len(add_edges)))
        hgraph.addEdges(add_edges)
    hgraph.print_stats()
    # why did loop end?
    if not hgraph.isCertifying():  # max card reached.
        # in-place on hgraph
        retval = grow_routine(hgraph, simpl, extents, max_ext, max_card, slice_func=slice_func, savefile=savefile, dbpath=args.dbpath, shapename=shapename)
    else:  # graph is certifying
        retval = extents, [c] * (start_ext + 1)
    return retval


def grow_routine(hgraph: Hypergraph, simpl, extents, max_ext, max_card, slice_func,
                 savefile:Savefile=None, dbpath=None, shapename="shape"):
    """

    :param hgraph:
    a hgraph which consists of a balanced pyramid/box where all valid hyperedges of a certain max cardinality are added.
    Caution: No edges will be created between only the vertices of this hgraph, we start by immediately shifting one side out
    :param simpl:
    :param extents: current extents
    :param max_ext:
    :param max_card:
    :return:
    """

    # current max_card for each extent (same max_card[ext] for each side of pyramid).
    # max_cards[1] = max_card for edges that include at least one vertex on shell of extent 1
    if not mit.all_equal(extents):
        vwarn(0, "All extents should be equal for better results")
        extents = max(extents) * len(extents)
    d = hgraph.d
    n_faces = len(extents)
    max_cards = {i : max_card for i in range(extents[0]+1)}

    ext = extents[0] # all extents are equal
    # start the grow loop:
    # STOP when graph is too big already
    # if no edges can be added (card. d+1 reached):
    # grow pyramide to next "level" (shift one wall out to a extent that was currently not seen)
    while ext < max_ext:
        ext += 1
        verts_on_ext = [set() for _ in range(n_faces)]
        verts_before = hgraph.vertices.copy()
        max_cards[ext] = 1

        # add a batch of hyperedges in current extent layer (probably: first card. 2, then card. 3, ... until args.max_cardinality)
        while max_cards[ext] < max_card:
            max_cards[ext] += 1
            vprint(1,"card {} (ext {})".format(max_cards[ext], ext))

            for add_side in range(n_faces):
                while extents[add_side] < ext:
                    vprint(1,"Adding a side in dir {} to {} with extents {}".format(add_side, shapename, extents))
                    extents[add_side] += 1
                    slice = slice_func(extents, add_side)
                    slice.difference_update(simpl.vertices)
                    vprint(4, extents)
                    vprint(4, slice)
                    verts_on_ext[add_side].update(slice)
                    hgraph.addVertices(slice)
                slice = verts_on_ext[add_side]

                # find edges between the slice on this face and all vertices that were added before.
                # so we make sure that no duplicates are checked
                verts_before_slice = verts_before.union(*verts_on_ext[:add_side])

                add_edges = []
                # attention: there was this code before, but I think it was nonsense: (commit e02a7f0d)
                # verts_before_slice = hgraph.vertices.difference(
                #     set().union(*slice_added_vertices[add_side:d+1]))

                # edges supported on slice have not to be checked
                # unless extent is 0
                if extents[add_side] == 0:
                    for edge in it.combinations(slice, max_cards[ext]):
                        if not issup(edge, hgraph.hyperedges) and simpl.isHyperedge(edge):
                            add_edges.append(edge)
                for s in range(1, max_cards[ext]):
                    for p1 in it.combinations(verts_before_slice, max_cards[ext] - s):
                        for p2 in it.combinations(slice, s):
                            edge = p1 + p2
                            # don't add edges that are supersets of already added edges
                            if not issup(edge, hgraph.hyperedges) and simpl.isHyperedge(edge):
                                add_edges.append(edge)
                vprint(1,"face {} on extent {} adds {} valid edges of card {}".format(
                    add_side, ext, len(add_edges), max_cards[ext]))
                bef = len(hgraph.hyperedges)
                hgraph.addEdges(add_edges)
                if len(hgraph.hyperedges)-bef != len(add_edges):
                    vwarn(f"Tried to add duplicate edges ({ext}, {max_cards}, {add_side})")

                if hgraph.isCertifying():
                    # graph is certifying and we can save it to a file simplex-*extent.obj
                    # proceed to ausduennen (there we can save it again or overwrite the old save)
                    print("Certifying with ext {} and max-cards {}".format(extents, max_cards))
                    if savefile:
                        savefile.extents = tuple(extents)
                        savefile.maxCard = max(max_cards.values())
                    return extents, max_cards

    if dbpath and savefile and question("Save too big graph?"):
        savefile.extents = tuple(extents)
        savefile.maxCard = max(max_cards.values())
        savefile.additional["tooBigException"] = True
        savefile.hypergraph = hgraph
        savefile_io.saveToDatabase(dbpath, savefile)
    print("Graph too big")
    return None


def dynamic_grow(hg: Hypergraph, simpl, args, savefile:Savefile=None, warmstart:Savefile=None):
    """
    to kill plot windows and loop: cmd+alt+shift+escape
    to zoom: right-click and scroll
    :param hg: empty hypergraph
    :param args:
    :return:
    """

    if warmstart:
        vprint(1, "warmstarting ...")
        savefile.warmStart = warmstart
        extents = warmstart.extents

        # this should work for this specific type of Einbettung (since all faces are just mirrored
        # copies of each other and we can keep the original colors on every lifted vertex)
        # here we assume that either the graph is certifying (k >= d+1) or that the
        # chromatic number is calculated and set exactly
        k = warmstart.chromNumber or (warmstart.dim + 1)

        def lift(verts:Collection, edges:Collection) -> Tuple[Set, Set]:
            """
            by one dimension
            :param verts:
            :param edges:
            :return:
            """
            if not verts:
                return set()
            if edges is None:
                edges = set()
            d = len(next(iter(verts)))
            ret_verts = set()
            ret_edges = set()
            # all straight faces
            for i in range(d+1):
                for v in verts:
                    nv = v[:i] + (0,) + v[i:]
                    ret_verts.add(nv)
                for e in edges:
                    ne = tuple(v[:i] + (0,) + v[i:] for v in e)
                    ret_edges.add(ne)
            # diagonal face
            for v in verts:
                nv = v + (1-sum(v),)
                ret_verts.add(nv)
            for e in edges:
                ne = tuple(v + (1-sum(v),) for v in e)
                ret_edges.add(ne)
            return ret_verts, ret_edges

        verts = set(warmstart.hypergraph.vertices)
        edges = set(warmstart.hypergraph.hyperedges)
        for _ in range(args.d-warmstart.hypergraph.d):
            verts, edges = lift(verts, edges)
        hg.vertices = verts
        hg.hyperedges = edges
        if args.debug:
            hg.plot_3d()

        extents = [-1] * (args.d + 1)
        altered = True
        while altered:
            altered = False
            for i in range(args.d + 1):
                extents[i] += 1
                cp = createPyramid(extents, args.d, flip=True)
                if cp.issubset(hg.vertices):
                    altered = True
                else:
                    extents[i] -= 1

    else:
        if args.extent is None:
            args.extent = 0

        print("Starting with a flip-pyramid of uniform extent",args.extent)
        hg.vertices = createPyramid(args.extent, args.d, flip=True)
        # no edges in graph, 1-coloring is possible
        k = 1

        extents = [args.extent] * (args.d+1)


    orig_verts = hg.vertices.copy()
    if orig_verts.issubset(createPyramid(extents, args.d, flip=True)):
        vprint(1, " (reached orig)")
        orig_verts = {"pla"}

    i = 0

    while True:
        print(f"Iteration #{i}", end="")
        i+=1
        if args.debug:
            hg.print_stats()
        color_dict = hg.getKColoring(k)
        while not color_dict:
            k += 1
            color_dict = hg.getKColoring(k)

        vprint(2, " k =", k, end="")

        if k >= args.d + 1:
            # found certifying graph
            break

        # sets = sorted(color_dict.values(), key=lambda s: len(s), reverse=True)
        # vertDegs = {v : sum(1 for h in hg.hyperedges if v in h) for v in hg.vertices}
        new_edges = []
        for colorclass in color_dict.values():
            # while not new_edges and len(sets):
            #     smallest = sets.pop()
            ne = simpl.findHyperedgeMinCoeff(colorclass)

            # ne = simpl.findHyperedgeMinCoeffMaxMinDeg(colorclass, vertDegs)
            # ne = simpl.findHyperedgeMinCoeffMaxDeg(colorclass, vertDegs)

            if ne:
                new_edges.append(ne)

        if len(new_edges):
            vprint(2,f"\nAdding {len(new_edges)} edges")
            hg.addEdges(new_edges)
        else:
            vprint(2,"\n"); hg.print_stats(v=2)

            vprint(1,"\nNo possible edges found. Adding some vertices...",end="")
            add_side = extents.index(min(extents))
            extents[add_side] += 1
            color_dict["new"] = pyramidSlice(extents, add_side, flip=True).difference(simpl.vertices)
            hg.vertices = hg.vertices.union(color_dict["new"])
            if orig_verts.issubset(hg.vertices):
                vprint(1, " (reached orig)",end="")
                orig_verts = {"pla"}
            vprint(1,"\n",end="")

        print("\r", end="")

        if args.debug:
            save_graph = hg.plot(color_dict = color_dict, highlight_edges=new_edges,
                                 iter=i, title=f"Iteration {i}")
            if save_graph:
                savefile_io.saveToDatabase(args.dbpath, Savefile(config=args.config,hypergraph=hg))

    vprint(1,"\nfinish dynamic")

    max_card = max(len(h) for h in hg.hyperedges)
    if savefile:
        savefile.iterations = i
        savefile.maxCard = max_card
        # savefile.chromNumber = args.d+1
    hg.print_stats()
    return extents, max_card