
# coding: utf-8

import argparse
import sys

# my scripts
import savefile_io
import utils
from analysis import Ausduennen
from hypergraph import Hypergraph, get_without_supersets
from savefile_io import Configuration, Savefile
from setup import *
from simplex import Simplex
from shapes import pyramidSlice, boxSlice, createPyramid
from algorithms import grow_starter, dynamic_grow
from utils import vprint, vwarn, question


def chooseShape():
    print("Choose shape:")
    return utils.chooseFromList(
        ["simpl", "simplflip", "box", "dyn"],
        lambda k:
        {"simpl": "Pyramid", "simplflip": "Flipped Pyramid",
         "box": "Boxed", "dyn": "Dynamic"}[k],
        allowCancel=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--shape", help="Shape: dyn, box, simpl[flip] (pyramide)", type=str)
    parser.add_argument("-d", help="Dimension", type=int)
    parser.add_argument("-e", "--extent", help="use a box or pyramide with extent [-extent, extent]^d", type=int)
    parser.add_argument("-c", "--max-cardinality", help="Max cardinality of hyperedges", type=int)
    parser.add_argument("--dbpath", help="Name of the database file", type=str)
    parser.add_argument("-n", help="Don't search in savefile and don't save to savefile.", action='store_true')
    parser.add_argument("--debug", help="Debug mode", action='store_true')
    parser.add_argument("-v", help="verbosity", type=int, default=DEFAULT_VERBOSITY)
    parser.add_argument("--delete", help="Delete a savefile", action='store_true')

    args = parser.parse_args()

    utils.VerbosityManager.global_verbosity = args.v

    if args.debug:
        vwarn("-- Debug mode is turned on!! --")

    if len(sys.argv) == 1:
        if utils.chooseFromList(["Load hypergraph", "New hypergraph"]) == "New hypergraph":
            args.shape = chooseShape()
            try: args.d = int(input("Dimension: "))
            except: pass
            try: args.max_cardinality = int(input("Max-cardinality ([enter] for default): "))
            except: pass

    config = Configuration(shape=args.shape,
                           dim=args.d,
                           startExtent=args.extent,
                           startMaxCard=args.max_cardinality)
    args.config = config

    if args.max_cardinality is not None and args.max_cardinality < 2:
        raise Exception("Please provide a max_cardinality >= 2 or remove option (default: d+1)")

    if args.n:
        config.additional["n"] = True

    if not args.dbpath:
        args.dbpath = DEFAULT_DATABASE
    if not os.path.exists(args.dbpath):
        args.dbpath = os.path.join(PATH_SAVEFILES, args.dbpath)
    if not os.path.exists(args.dbpath):
        print(f"Database {args.dbpath} not found")

    sf = None
    hgraph = None
    if args.delete:
        if args.n:
            vwarn("Remove -n option")
            assert False
        print("delete")
        sf = savefile_io.deleteFromDatabase(args.dbpath, config)
        if not sf:
            return
        print("Successfully deleted!")
        if not question("Continue with popped graph?", False):
            return
        hgraph = sf.hypergraph
    else:
        if not args.n:
            print("Saved matches:")
            sf = savefile_io.findInDatabase(args.dbpath, config)
        if not sf:
            if not args.shape or not args.d:
                if question("Start new hypergraph?", defaultYes=False):
                    if not args.shape:
                        args.shape = chooseShape()
                        if not args.shape:
                            return
                    if not args.d:
                        try:
                            args.d = int(input("Dimension: "))
                        except:
                            print("Invalid dimension")
                            return
                    if args.d and args.shape and not args.max_cardinality:
                        try:
                            args.max_cardinality = int(input("Max-cardinality ([enter] for default): "))
                        except:
                            pass
                else: return
            sf = Savefile(config=config)
            sf.updateTimestamp()
        hgraph = sf.hypergraph

    if sf and not args.d:
        args.d = sf.dim

    simpl = Simplex(args.d)

    # if hgraph was not found in database
    if not hgraph:
        print("Not found in database")

        start_time = time.time()

        if args.shape == "dyn":
            # create an empty hypergraph
            hgraph = Hypergraph(args.d)
            ws = None
            if question("Warmstart with graph from db?"):
                ws = savefile_io.findInDatabase(dbpath=args.dbpath,
                                                conf=savefile_io.Configuration(dim=args.d-1))
            retval = dynamic_grow(hgraph, simpl, args, sf, warmstart=ws)


        elif args.shape.startswith("simpl"):
            # create an empty hypergraph
            hgraph = Hypergraph(args.d)

            flip = args.shape.endswith("flip")

            # set parameters
            max_card = args.max_cardinality or float("inf")
            max_card = min(max_card, args.d + 1)
            start_ext = args.extent or 0
            max_ext = 6

            retval = grow_starter(hgraph=hgraph, start_ext=start_ext,
                                  max_ext=max_ext, max_card=max_card, n_facets=args.d + 1,
                                  slice_func=lambda e, a: pyramidSlice(e, added_side=a, flip=flip),
                                  base_shape_func=lambda e, d: createPyramid(e, d, flip=flip),
                                  simpl=simpl, shapename="flipped pyramide" if flip else "pyramide",
                                  args=args, savefile=sf)

        elif args.shape == "box":
            # create an empty hypergraph
            hgraph = Hypergraph(args.d)

            # set parameters
            max_card = args.max_cardinality or float("inf")
            max_card = min(max_card, args.d + 1)
            start_ext = args.extent or 0
            max_ext = 6

            retval = grow_starter(hgraph=hgraph, start_ext=start_ext,
                                  max_ext=max_ext, max_card=max_card, n_facets=2 * args.d,
                                  slice_func=boxSlice,
                                  base_shape_func=lambda e, d: set(
                                    it.product(range(-e, e + 2), repeat=d)).difference(simpl.vertices),
                                  simpl=simpl, shapename="box", args=args, savefile=sf)

        if hgraph:
            exectime = time.time() - start_time
            sf.execTime = exectime
            wording = "pyramid" if args.shape.startswith("simpl") else "dynamic graph" if args.shape == "dyn" else "box"
            if retval:
                sf.certifying = True
                print("--- certifying {} found: {:.2f} seconds ---\n".format(wording, exectime))
            else:
                sf.certifying = False
                print("--- no certifying {} found (max ext: {}): {:.2f} seconds ---\n".format(wording, max_ext, exectime))

    # what we need here: hgraph and args.dbpath

    if hgraph:
        if not hgraph.isCertifying():
            chr_n = hgraph.calcChromaticNumber(start_at=args.d)
            sf.chromNumber = chr_n
            print("The chromatic number (d={}) is: {}".format(args.d, chr_n))
        else:
            if any(len(e) == 1 for e in hgraph.hyperedges) or (simpl.vertices & hgraph.vertices):
                print("Watch out! There are hyperedges of card 1 or simplex vertices in your graph!")
            hgraph.print_stats()
            print("The chromatic number (d={}) is at least: {}".format(args.d, args.d+1))
            print()
            # if input("Stable Set Analysis? (y,[n])") == "y":
            #     print("Stable Set Analysis")
            #     stable_set_analysis(hgraph)

            if input("ausduennen? (y,[n]) ") == "y":
                ausduennen_loop(args, hgraph, sf)


def ausduennen_loop(args, hgraph, sf):
    before = len(hgraph)
    hgraph = Hypergraph(args.d, get_without_supersets(hgraph.hyperedges))
    if len(hgraph) != before:
        print("Deleted {} superedges. Left with {} edges".format(before - len(hgraph), len(hgraph)))
    hg_copy = hgraph.copy()
    thinoutSeq = []
    parent = None
    while True:
        print("+++")
        hg_copy.print_stats()

        menu_dict = \
            {
                ("v", "11", "12"): ("Knoten nach aufsteigendem grad loeschen", Ausduennen.verticesAsc),
                ("V",): ("Knoten nach absteigendem grad loeschen", Ausduennen.verticesDesc),
                ("e",): ("Hyperkanten nach absteigender cardinality loeschen", Ausduennen.edgesDesc),
                ("E",): ("Hyperkanten nach aufsteigender cardinality loeschen", Ausduennen.edgesAsc),
                ("g",): ("random greedy Hyperkanten loeschen", Ausduennen.randomEdges),
                ("G",): ("random greedy Knoten loeschen", Ausduennen.randomVertices),
                ("s",): ("supersets entfernen", Ausduennen.withoutSupersets),
                ("a",): ("stable set analysis", Ausduennen.stableSetAnalysis),
                ("p",): ("ausgeben", None),
                ("f",): ("speichern", None),
                ("r",): ("reset", None),
                ("q", "c"): ("abbrechen", None)
            }

        print("Methode:")
        for i, k in enumerate(menu_dict.keys()):
            print(f" {'(' + str(i + 1) + ')':>4} {menu_dict[k][0]}")
        method = input("Eingabe: ")

        key = None
        try:
            key = list(menu_dict.keys())[int(method) - 1]
        except:
            try:
                # get first key for chosen method
                key = [k for k in menu_dict.keys() if method in k][0]
            except:
                key = method

        if isinstance(key, tuple) and menu_dict[key][1]:
            try:
                info = len(hg_copy.vertices), len(hg_copy.hyperedges)

                # run method
                menu_dict[key][1](hg_copy)

                if info != (len(hg_copy.vertices), len(hg_copy.hyperedges)):
                    thinoutSeq.append(key[0])
            except KeyboardInterrupt as e:
                pass
        else:
            # reset graph
            if "r" in key:
                hg_copy = hgraph.copy()
                thinoutSeq = []

            # end
            elif "q" in key:
                break

            # print/plot
            elif "p" in key:
                # sf.populateWithGraph(hg_copy)
                hg_copy.plot(title=sf.strQuick(), color_dict=hg_copy.getKColoring(hg_copy.d+1))
            elif "pp" in key:
                print(hg_copy.hyperedges)


            # save
            elif "f" in key:
                # fname = f"AD-{args.shape}-{args.d}-{args.extent}-" \
                #         f"{args.max_cardinality}-({len(vertices)},{len(hg_copy.hyperedges)})"
                # s = Savefile(**config.__dict__)
                if args.n and (not args.dbpath or not question(f"Override args.n? (dbpath={args.dbpath})", defaultYes=True)):
                    continue
                sf.thinoutSeq = thinoutSeq.copy()
                sf.populateWithGraph(hg_copy)
                if sf.certifying: vprint(2, "Is certifying! (checked before saving)")
                else: vprint(2, "Is not certifying! (checked before saving)")
                sf.parent = parent
                if question("Is it minimal?", defaultYes=True):
                    sf.isMinimal = True
                else:
                    sf.isMinimal = False
                savefile_io.saveToDatabase(args.dbpath, sf)
                parent = sf.copy()

            elif "bb" in key:
                breakpoint()

            elif "aa" in key:
                Ausduennen.stableSetAnalysisWeak(hg_copy)


if __name__ == "__main__":
    main()