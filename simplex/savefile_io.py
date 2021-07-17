from __future__ import annotations
import jsonpickle
from datetime import datetime

from setup import *
import utils
from hypergraph import Hypergraph

HGFOLDER = "hypergraphs"

class Configuration:
    """
    Baseclass for Savefile. Used to define the base properties of savefiles when searching for some.
    Implements methods to compare savefiles.
    """
    def __init__(self, shape:str=None, dim:int=None, startExtent:int=None, startMaxCard:int=None, additional:dict={}):
        """

        :param shape:
        :param dim:
        :param startExtent:
        :param startMaxCard:
        :param additional:
        """
        self.shape = shape
        self.dim = dim
        self.startExtent = startExtent
        self.startMaxCard = startMaxCard
        self.additional = additional

    def matches(self, other):
        return all((self.shape        == other.shape,
                    self.dim          == other.dim,
                    self.startExtent  == other.startExtent,
                    self.startMaxCard == other.startMaxCard))

    def compatible(self, other):
        return all((other.shape        is None or self.shape        == other.shape,
                    other.dim          is None or self.dim          == other.dim,
                    other.startExtent  is None or self.startExtent  == other.startExtent,
                    other.startMaxCard is None or self.startMaxCard == other.startMaxCard)) \
               or \
               all((self.shape is None or self.shape == other.shape,
                    self.dim is None or self.dim == other.dim,
                    self.startExtent is None or self.startExtent == other.startExtent,
                    self.startMaxCard is None or self.startMaxCard == other.startMaxCard))

    def strQuick(self):
        return f"{self.shape}, dim={self.dim}, e={self.startExtent}, c={self.startMaxCard}"


class Savefile(Configuration):
    def __init__(self,
                 shape:str=None, dim:int=None, startExtent:int=None, startMaxCard:int=None, additional:dict=None,
                 fname:str=None,
                 extents:Tuple[int, ...]=None, maxCard:int=None, iterations:int=None, execTime:float=None,
                 warmStart:Savefile=None,
                 nEdges:int=None, edgesCard:dict=None, nVertices:int=None, verticesDegree:dict=None,
                 certifying:bool=None, chromNumber:int=None, isMinimal:bool=None,
                 thinoutSeq:List=None, parent:Savefile=None,
                 creationTimestamp:str=None,
                 hypergraph:Hypergraph=None,
                 config:Configuration=None,
                 id:int=None):
        self.shape = shape
        self.dim = dim
        self.startExtent = startExtent
        self.startMaxCard = startMaxCard
        self.additional = additional
        self.fname = fname
        self.extents = extents
        self.maxCard = maxCard
        self.iterations = iterations
        self.execTime = execTime
        self.warmStart = warmStart
        self.nEdges = nEdges
        self.edgesCard = edgesCard
        self.nVertices = nVertices
        self.verticesDegree = verticesDegree
        self.certifying = certifying
        self.chromNumber = chromNumber
        self.isMinimal = isMinimal
        self.thinoutSeq = thinoutSeq
        self.parent = parent
        self.creationTimestamp = creationTimestamp
        self.id = id
        if config:
            for k, v in config.__dict__.items():
                if not self.__dict__[k]:
                    self.__dict__[k] = v
        self.populateWithGraph(hypergraph)

    def __str__(self):
        return str(self.__dict__)

    def strQuick(self):
        ret = super().strQuick()
        if self.thinoutSeq:
            ret += f", seq={''.join(self.thinoutSeq)}"
        ret += f" ({self.nVertices}, {self.nEdges}) id:{self.id}"
        if self.isMinimal is False:
            ret += " not min"
        if self.certifying is False:
            ret += " not cert"
        return ret

    def copy(self):
        return Savefile(**self.__dict__)

    def populateWithGraph(self, hgraph:Hypergraph):
        if not hgraph:
            self.hypergraph = None
            return
        self.hypergraph = hgraph.copy()
        self.nEdges = len(hgraph.hyperedges)
        self.edgesCard = hgraph.getHyperedgesStats()
        self.nVertices = len(hgraph.vertices)
        self.verticesDegree = hgraph.getVerticesStats()
        self.certifying = hgraph.isCertifying()
        if not self.dim:
            self.dim = hgraph.d
        if not self.certifying:
            self.chromNumber = hgraph.calcChromaticNumber(start_at=self.dim)
        else:
            self.chromNumber = self.dim + 1

    def withHypergraph(self, hgfolder):
        ret = self.copy()
        hgpath = os.path.join(hgfolder, ret.fname)
        try:
            with open(hgpath) as f:
                ret.hypergraph = jsonpickle.decode(f.read())
        except FileNotFoundError as e:
            print(f"Hypergraph under {hgpath} not found!")
        except Exception as e:
            print(f"Error reading Hypergraph in {hgpath}")
            vprint(2, e)
        return ret

    def withoutHypergraph(self):
        ret = self.copy()
        ret.hypergraph = None
        return ret

    def updateTimestamp(self):
        self.creationTimestamp = datetime.now().isoformat()


class Database(list):
    """
    Class to find savefiles in a list by their config instance.
    Also loads hypergraph into returned object.
    """

    def __init__(self, listSavefiles:List=[]):
        self.id_counter = 1
        self.extend(listSavefiles)

    def __bool__(self):
        #Database is a list, just to make sure that nobody tries "if not db" and forgets it could have length 0
        return True

    def setIds(self):
        if not hasattr(self, "id_counter"):
            self.id_counter = 1
        for obj in self:
            if obj.id is None:
                obj.id = self.id_counter
                self.id_counter += 1

    def get(self, conf:Configuration, delete=False):
        if not conf:
            return None
        for sf in self:
            if sf.matches(conf):
                if delete:
                    self.remove(sf)
                return sf.withHypergraph(self.hgfolder)
        return None

    def choose(self, conf:Configuration=None, delete=False, choose: int = None):
        if not conf:
            conf = Configuration()
        compatibles = [sf for sf in self if sf.compatible(conf)]
        if not compatibles:
            return None
        if choose != None:
            if 0 <= choose < len(compatibles):
                return compatibles[choose].withHypergraph(self.hgfolder)
            return None
        ret = utils.chooseFromList(compatibles, strFunc=lambda s: s.strQuick(), allowCancel=True)
        if not ret:
            return 0
        if delete:
            self.remove(ret)
        return ret.withHypergraph(self.hgfolder)

def listDatabase(dbpath:str, conf:Configuration=None):
    """
    List all savefiles in this database.
    :param dbpath:
    :param conf:
    :return:
    """
    if not os.path.exists(dbpath):
        return
    db = getDatabase(dbpath)
    if not conf:
        conf = Configuration()
    compatibles = [sf for sf in db if sf.compatible(conf)]
    utils.printList(compatibles, strFunc=lambda s: s.strQuick())


def findInDatabase(dbpath: str, conf: Configuration = None, delete=False,
                   choose: int = None, id: int = None):
    """
    Find a savefile that is compatible with the passed configuration. User can choose out of the list by interactively
    typing the index or by passing it to this method.
    :param dbpath:
    :param conf: If None, every savefile in dbpath-database is listed
    :param delete: chosen savefile and linked hypergraph file will be deleted
    :param choose: pass a index which will be chosen
    :param id: pass an id which will be chosen
    :return:
    """
    if not os.path.exists(dbpath):
        return

    db = getDatabase(dbpath)

    if id != None:
        try:
            return next(r for r in db if hasattr(r, "id") and r.id == id).withHypergraph(db.hgfolder)
        except:
            vprint(1, f"Savefile for id {id} not found.")
            return None

    ret = db.choose(conf, delete=delete, choose = choose)
    if ret is None:
        vprint(1, "Desired savefile not found.")
        return None
    elif ret == 0:
        return None
    else:
        vprint(1, "Loaded from database.")
    if delete and ret:
        if os.path.exists(os.path.join(db.hgfolder, ret.fname)):
            if all(ret.fname != sf.fname for sf in db):
                vprint(2, "Removing hypergraph file")
                os.remove(os.path.join(db.hgfolder, ret.fname))
        saveDatabase(dbpath, db)
    return ret

def deleteFromDatabase(dbpath: str, conf: Configuration):
    return findInDatabase(dbpath, conf, delete=True)

def saveToDatabase(dbpath: str, savefile: Savefile):
    """
    Saves savefile.hypergraph into a new file with appropriate name.
    That path is saved in savefile, while the hypergraph object is deleted from it (on a copy).
    The hypergraph object is deleted from all parents in the tree, whilst this one is not saved to a file.
    savefile is then added to database.
    :param savefile:
    :param dbpath:
    :return:
    """
    if not os.path.exists(dbpath):
        print(f"Creating new database at {dbpath}.")
        saveDatabase(dbpath, Database())
    db = getDatabase(dbpath)

    # save Hypergraph
    seq = savefile.thinoutSeq
    if seq == []:
        seq = ['0']
    if not seq:
        seq = []
    fname = f"hg-d{savefile.dim}-s{savefile.shape}-e{savefile.startExtent}-" \
            f"c{savefile.startMaxCard}-seq{''.join(seq)}-" \
            f"v{savefile.nVertices}-e{savefile.nEdges}.json"

    hgfolder=os.path.join(os.path.dirname(dbpath), HGFOLDER)
    if not os.path.exists(hgfolder):
        os.mkdir(hgfolder)

    i = 0
    hgpath=os.path.join(hgfolder, fname)
    # if hgraph savefile exists already, check if it has the equal content. If not,
    # add a incremential "-%d" at the end of the filename
    while os.path.exists(hgpath):
        with open(hgpath, 'r') as f:
            if f.read() == jsonpickle.encode(savefile.hypergraph):
                break
        i += 1
        fname = fname[:-5 if i == 1 else -6-len(str(i-1))] + f"-{i}" + ".json"
        hgpath = os.path.join(hgfolder, fname)

    with open(hgpath, 'w') as f:
        out = jsonpickle.encode(savefile.hypergraph)
        f.write(out)

    # save Savefile (without hypergraph)
    save = savefile.withoutHypergraph()
    save.fname = fname

    # remove hypergraph object from all parents
    def rem_hypergraphs(s:Savefile):
        if s.parent:
            s.parent = s.parent.withoutHypergraph()
            rem_hypergraphs(s.parent)
        if s.warmStart:
            s.warmStart = s.warmStart.withoutHypergraph()
            rem_hypergraphs(s.warmStart)

    rem_hypergraphs(save)

    save.id = None

    db.append(save)
    ret = saveDatabase(dbpath, db)
    if ret:
        vprint(1, f"Saved successfully! (id: {save.id})")
    else:
        vprint(1, "Error while saving...")
    return ret



def saveDatabase(dbpath: str, db: Database):
    """
    Saves Database to file (overrides file contents).
    id of savefiles is set if None.
    hgfolder attribute is deleted in order to make databases realtive to the folder they are in.
    Other than the id and hgdolder, db is taken as it is, hence hypergraph should
    already be set to None in every Savefile in database.
    :param db:
    :param dbpath: as full path
    :return: bool
    """
    if not db:
        print("No Database given.")
        return False

    try:
        del db.hgfolder
    except: pass

    db.setIds()

    try:
        with open(dbpath, 'w') as f:
            out = jsonpickle.encode(db, indent="  ")
            f.write(out)
            vprint(3, f"written database with {len(db)} savefiles to {dbpath}")
    except OSError as err:
        print(f"(save_to_file) OS error: {err}")
    except Exception as err:
        print(f"jsonpickle load error: {err}")
    else:
        return True
    return False

def getDatabase(dbpath):
    """
    Loads Datanabase from specified (full) path and sets hgfolder attribute appropriately.
    :param dbpath:
    :return: database
    """
    try:
        with open(dbpath, 'r') as f:
            db = jsonpickle.decode(f.read())
            hgfolder = os.path.join(os.path.dirname(dbpath), HGFOLDER)
            db.hgfolder = hgfolder
            return db
    except OSError as err:
        print(f"(get_from_file) OS error: {err}")
    except Exception as err:
        print(f"jsonpickle load error: {err}")
