from setup import *
from simplex import Simplex


def flipverts(verts:set):
    verts = np.array(list(verts))
    verts *= -1
    verts += 1
    return {tuple(p) for p in verts}


def flipsimpl(simplverts:set, extents:list):
    """
    make a regular simplex-like pyramid into a flipped one. Shift out diagonal facet by d-1 steps and then
    flip all coordinates as usual. It is taken care of the unit vectors and origin (simplex-vertices).
    :param simplverts:
    :param extents:
    :return:
    """
    d = len(extents)-1
    e = extents.copy()
    for _ in range(d-1):
        e[0] += 1
        simplverts = simplverts.union(pyramidSlice(e, 0))
    sverts = Simplex.getVertices(d)
    return flipverts(simplverts.union(sverts)).difference(sverts)


def pyramidSlice(extents: List[int], added_side: int, flip=False): #-> Set[Tuple[int, ...],...]:
    """
    includes simplex vertices!
    :param extents: list of pyramid extents
    :param added_side: index of side that grew by 1 extent, as in extents (0: diagonal, 1: in x_1, ...)
    :return: set of tuples (vertices)
    """
    def rec_diag(d, j=-1, x=None):
        # if j == -1 or not x:
        #   return rec_diag(d, 1, [])
        if j == d:
            return [x+[1 - sum(x) + extents[0]]]
        else:
            dret = []
            for x_j in range(-extents[j], 2 - sum(x) + sum(extents[j+1:]) + extents[0]):
                dret.extend(rec_diag(d,j+1,x+[x_j]))
            return dret

    d = len(extents)-1
    if flip:
        extents[0] += d - 1

    if added_side == 0:
        ret = [tuple(v) for v in rec_diag(d, 1, [])]
    else:
        borders = [range(-extents[j], 2 + sum(extents) - extents[j]) for j in range(1,d+1)]
        borders[added_side - 1] = [-extents[added_side]]
        ret = it.product(*borders)
        ret = [pt for pt in ret if sum(pt) <= 1 + extents[0]]

    if flip:
        extents[0] -= d - 1
        return flipverts(ret)
    else:
        return set(ret)


def boxSlice(extents: List[int], added_side: int):
    assert len(extents) % 2 == 0
    d = int(len(extents) / 2)
    borders = [range(-extents[j], 2 + extents[j + 1]) for j in range(0, 2 * d, 2)]
    borders[added_side // 2] = [-extents[added_side]] \
        if added_side % 2 == 0 else [1 + extents[added_side]]
    return set(it.product(*borders))


def createPyramid(ext, d, flip=False):
    """
    doesn't include simplex vertices!
    :param ext:
    :param d:
    :param flip:
    :return:
    """
    if isinstance(ext, int):
        ext = [ext] * (d+1)
    simpl = Simplex(d)
    vertices = set()
    extents = [-1] * (d + 1)
    for i in range(d + 1):
        for j in range(ext[i] + 1):
            extents[i] += 1
            # union vertices
            if flip:
                extents[0] += d - 1
            slice = pyramidSlice(extents, added_side=i)
            if flip:
                extents[0] -= d - 1
                slice = flipverts(slice)
            if j == 0:
                slice.difference_update(simpl.vertices)
            vertices.update(slice)
    return vertices