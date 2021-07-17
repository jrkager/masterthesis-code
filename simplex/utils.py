import time
import sty
import os
import math

### VERBOSITY ###
class VerbosityManager:
	"""
	utility class that stores the global verbosity values so that they can be used by every other file
	"""
	# 0: don't print anything but input-dialogs
	# 1: only print significant results and process stats
	# 2: print more
	# 3: print even more
	# 4: debug mode
	global_verbosity = -1

def vprint(min_verbosity=None, *args, **kwargs):
    """
    utility function that works like print() but takes an additional first parameter which indicates the minimum
    level of verbosity for printing this. if verbosity is less than min_verbosity nothing is printed.
    :param min_verbosity:
    :param args: just like standard function print
    :param kwargs: just like print
    :return:
    """
    if min_verbosity is None:
        print(*args, **kwargs)
    elif not isinstance(min_verbosity, int):
        print(min_verbosity, *args, **kwargs)
    elif VerbosityManager.global_verbosity >= min_verbosity:
        print(*args, **kwargs)

def vwarn(min_verbosity=None, *args, **kwargs):
    """
    like vprint(), but prints a yellow "Warning: " before the text.
    """
    warn_text = sty.fg.yellow + "Warning:" + sty.fg.rs
    if min_verbosity is None:
        print(warn_text, *args, **kwargs)
    elif not isinstance(min_verbosity, int):
        print(warn_text, min_verbosity, *args, **kwargs)
    elif VerbosityManager.global_verbosity >= min_verbosity:
        print(warn_text, *args, **kwargs)

def toTuple(v):
    if v is None:
        return v
    try: return tuple(v)
    except: return tuple([v])

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f s' % \
            (method.__name__, te - ts))
        return result
    return timed


def set_compare(s1, s2):
    print("Sets are equal:", s1==s2)
    if not s1 == s2:
        print("Lengths:", len(s1),len(s2))
        print("only left", s1.difference(s2))
        print("only right", s2.difference(s1))


def print_progress(i, N):
    print(f"{i / N * 100 :5.1f}%", end="\r")


def printList(objlist, strFunc=None):
    if not strFunc:
        strFunc = str
    digits = len(str(len(objlist)+1))
    for i, obj in enumerate(objlist):
        print(f"{i:>{digits + 1}}) {strFunc(obj)}")

def chooseFromList(objlist, strFunc=None, allowCancel=False):
    digits = len(str(len(objlist)+1))
    try:
        printList(objlist, strFunc)
        if allowCancel:
            print(f"{'x':>{digits+1}}) cancel")
        choice = input("Choice: ")
        if choice == "x":
            return None
        return objlist[int(choice)]
    except (IndexError, ValueError) as _:
        return chooseFromList(objlist, strFunc, allowCancel)

def question(text, defaultYes=False):
    if defaultYes:
        return not input(text + " ([y],n) ").lower() in ("n","c")
    return input(text + " (y,[n]) ").lower() == "y"


def figsize(scale=1,scale_height=None, ratio=None):
    fig_width_pt = 360.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (math.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    if not ratio:
        ratio = golden_mean
    if scale_height:
        ratio *= scale_height
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def savefig(plt, filename):
    plt.savefig(os.path.join('../plots/','{}.pgf'.format(filename)))
    plt.savefig(os.path.join('../plots/','{}_pdf.pdf'.format(filename)))