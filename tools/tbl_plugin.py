"""pytest plugin: which tests look up which keys of sqlglot's module-level tables.

Used only to build app/tbl.json.gz, never at scoring time. Every non-empty
dict display bound to one name at module or class level in sqlglot is
compiled as a TDict, a dict subclass that knows which displays ("sites") its
entries came from. Copies (`.copy()`, `{**T, ...}` displays) keep the sites
of their sources, so a dialect's table built from the generic one still
reports the generic site. Per test, each test in its own fork of the
collected session (so run-once caches cannot hide a reader), it records:

  - look: (site, key name) for every keyed read (`[]`, get, in, pop,
    setdefault) on any table carrying that site, hit or miss
  - all:  sites of tables read as a whole (iteration, keys, items, values,
    comparison, repr, pickling, `|`)

Import-time reads go to their own bucket, so the selector can see when a
table feeds other import-time state. For every site it also records each
entry's line span and the names of the keys that entry produced.

Key names: a class by __name__, a str as "s:" + value, anything else
"r:" + repr.

  docker run ... -e TBL_OUT=/out/tbl0.json -e TBL_ROOT=/opt/sqlglot \\
    -e PYTHONPATH=/tools:/opt/sqlglot -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
    <image> python -m pytest -q -p tbl_plugin -p no:cacheprovider -o addopts= <files>
"""

from __future__ import annotations

import ast
import concurrent.futures
import importlib.abc
import importlib.machinery
import json
import os
import pickle
import sys

ROOT = os.path.realpath(os.environ.get("TBL_ROOT", os.getcwd())) + os.sep
PKG = ROOT + "sqlglot" + os.sep

LOOK: set = set()
ALL: set = set()
ENTRIES: dict = {}   # site -> {(lo, hi): set(names)}
FILES: list = []
SETS = os.environ.get('TBL_SETS', '1') == '1'
WHO = {} if os.environ.get('TBL_WHO') else None
BUCKETS: dict = {}   # bucket key -> {"look": set, "all": set}


def kname(k) -> str:
    if isinstance(k, type):
        return k.__name__
    if isinstance(k, str):
        return "s:" + k
    try:
        return "r:" + repr(k)
    except Exception:  # noqa: BLE001
        return "r:?"


def _who(sites):
    f = sys._getframe(2)
    while f is not None and "tbl_plugin" in f.f_code.co_filename:
        f = f.f_back
    if f is not None:
        w = f"{f.f_code.co_filename.replace(ROOT, '')}:{f.f_lineno}:{f.f_code.co_name}"
        for st in sites:
            WHO.setdefault(st, set()).add(w)


class TDict(dict):
    __slots__ = ("_sites",)

    def _r(self, k):
        try:
            n = kname(k)
        except Exception:  # noqa: BLE001
            self._all()
            return
        for s in self._sites:
            LOOK.add((s, n))

    def _all(self):
        ALL.update(self._sites)
        if WHO is not None:
            _who(self._sites)

    def __getitem__(self, k):
        self._r(k)
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        self._r(k)
        return dict.get(self, k, default)

    def __contains__(self, k):
        self._r(k)
        return dict.__contains__(self, k)

    def setdefault(self, k, default=None):
        self._r(k)
        return dict.setdefault(self, k, default)

    def pop(self, k, *a):
        self._r(k)
        return dict.pop(self, k, *a)

    def popitem(self):
        self._all()
        return dict.popitem(self)

    def copy(self):
        o = TDict(dict.items(self))
        o._sites = self._sites
        return o

    def __copy__(self):
        return self.copy()

    def keys(self):
        self._all()
        return dict.keys(self)

    def items(self):
        self._all()
        return dict.items(self)

    def values(self):
        self._all()
        return dict.values(self)

    def __iter__(self):
        self._all()
        return dict.__iter__(self)

    def __reversed__(self):
        self._all()
        return dict.__reversed__(self)

    def __eq__(self, other):
        self._all()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        self._all()
        return dict.__ne__(self, other)

    __hash__ = None

    def __len__(self):
        self._all()
        return dict.__len__(self)

    def __bool__(self):
        self._all()
        return dict.__len__(self) != 0

    def __or__(self, other):
        self._all()
        return dict.__or__(self, other)

    def __ror__(self, other):
        self._all()
        return dict.__ror__(self, other)

    def __repr__(self):
        self._all()
        return dict.__repr__(self)

    __str__ = __repr__

    def __reduce_ex__(self, proto):
        self._all()
        return (dict, (dict(dict.items(self)),))

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __format__(self, spec):
        self._all()
        return dict.__format__(self, spec)


def __tbl__(site, entries):
    o = TDict()
    sites = {site}
    rec = ENTRIES.setdefault(site, {})
    for lo, hi, unpack, k, v in entries:
        names = rec.setdefault((lo, hi), set())
        if unpack:
            if isinstance(v, TDict):
                sites |= v._sites
                pairs = list(dict.items(v))
            elif isinstance(v, dict):
                pairs = list(dict.items(v))
            else:
                pairs = [(kk, v[kk]) for kk in v.keys()]
            for kk, vv in pairs:
                names.add(kname(kk))
                dict.__setitem__(o, kk, vv)
        else:
            names.add(kname(k))
            dict.__setitem__(o, k, v)
    o._sites = frozenset(sites)
    return o


def _sset(x):
    """(plain data, sites) of a set operand, without logging a read."""
    if isinstance(x, TSet):
        return x._d, x._sites
    return x, frozenset()


class TSet:
    """Stands in for a class or module level set. Not a set subclass, so no
    C fast path can read it unseen: every route is a Python method here.
    Membership is a keyed read. Pointwise operations (|, &, -, ^, union,
    copy...) hand the sites on to the result. Anything else reads it whole."""

    __slots__ = ("_d", "_sites")

    def __init__(self, data, sites):
        self._d = data
        self._sites = sites

    @property
    def __class__(self):  # isinstance(x, set) keeps answering as before
        return type(self._d)

    def _r(self, k):
        try:
            n = kname(k)
        except Exception:  # noqa: BLE001
            self._all()
            return
        for s in self._sites:
            LOOK.add((s, n))

    def _all(self):
        ALL.update(self._sites)
        if WHO is not None:
            _who(self._sites)

    def __contains__(self, k):
        self._r(k)
        return k in self._d

    def __iter__(self):
        self._all()
        return iter(self._d)

    def __len__(self):
        self._all()
        return len(self._d)

    def __bool__(self):
        self._all()
        return bool(self._d)

    def __repr__(self):
        self._all()
        return repr(self._d)

    __str__ = __repr__

    def __format__(self, spec):
        self._all()
        return format(self._d, spec)

    def __hash__(self):
        self._all()
        return hash(self._d)

    def __reduce_ex__(self, proto):
        self._all()
        return (type(self._d), (list(self._d),))

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __copy__(self):
        return TSet(self._d.copy(), self._sites)

    copy = __copy__

    def __deepcopy__(self, memo):
        self._all()
        import copy as _c
        return _c.deepcopy(self._d, memo)

    def _cmp(name):  # noqa: N805
        def f(self, other):
            self._all()
            d, _ = _sset(other)
            if isinstance(other, TSet):
                other._all()
            return getattr(self._d, name)(d)
        f.__name__ = name
        return f

    __eq__ = _cmp("__eq__")
    __ne__ = _cmp("__ne__")
    __le__ = _cmp("__le__")
    __lt__ = _cmp("__lt__")
    __ge__ = _cmp("__ge__")
    __gt__ = _cmp("__gt__")
    issubset = _cmp("issubset")
    issuperset = _cmp("issuperset")
    isdisjoint = _cmp("isdisjoint")
    del _cmp

    def _bin(name):  # noqa: N805
        def f(self, other):
            d, st = _sset(other)
            if not isinstance(d, (set, frozenset)):
                return NotImplemented
            return TSet(getattr(self._d, name)(d), self._sites | st)
        f.__name__ = name
        return f

    __or__ = _bin("__or__")
    __and__ = _bin("__and__")
    __sub__ = _bin("__sub__")
    __xor__ = _bin("__xor__")
    __ror__ = _bin("__ror__")
    __rand__ = _bin("__rand__")
    __rsub__ = _bin("__rsub__")
    __rxor__ = _bin("__rxor__")
    del _bin

    def _nary(name):  # noqa: N805
        def f(self, *others):
            sites = self._sites
            ds = []
            for o in others:
                d, st = _sset(o)
                sites = sites | st
                ds.append(d)
            return TSet(getattr(self._d, name)(*ds), sites)
        f.__name__ = name
        return f

    union = _nary("union")
    intersection = _nary("intersection")
    difference = _nary("difference")
    symmetric_difference = _nary("symmetric_difference")
    del _nary

    def _mut(name):  # noqa: N805
        def f(self, *others):
            ds = []
            for o in others:
                d, st = _sset(o)
                self._sites = self._sites | st
                ds.append(d)
            getattr(self._d, name)(*ds)
        f.__name__ = name
        return f

    update = _mut("update")
    intersection_update = _mut("intersection_update")
    difference_update = _mut("difference_update")
    symmetric_difference_update = _mut("symmetric_difference_update")
    del _mut

    def _imut(name):  # noqa: N805
        def f(self, other):
            d, st = _sset(other)
            if not isinstance(d, (set, frozenset)):
                return NotImplemented
            self._sites = self._sites | st
            self._d = getattr(self._d, name)(d)
            return self
        f.__name__ = name
        return f

    __ior__ = _imut("__or__")
    __iand__ = _imut("__and__")
    __isub__ = _imut("__sub__")
    __ixor__ = _imut("__xor__")
    del _imut

    def add(self, k):
        self._d.add(k)

    def discard(self, k):
        self._r(k)
        self._d.discard(k)

    def remove(self, k):
        self._r(k)
        self._d.remove(k)

    def pop(self):
        self._all()
        return self._d.pop()

    def clear(self):
        self._d.clear()

    def __getattr__(self, name):  # any other route reads the table whole
        self._all()
        return getattr(self._d, name)


def __tbls__(site, entries):
    out = set()
    sites = {site}
    rec = ENTRIES.setdefault(site, {})
    for lo, hi, star, v in entries:
        names = rec.setdefault((lo, hi), set())
        if star:
            if isinstance(v, TSet):
                sites |= v._sites
                v = v._d
            elif isinstance(v, TDict):
                sites |= v._sites
                v = dict.keys(v)
            for kk in v:
                names.add(kname(kk))
                out.add(kk)
        else:
            names.add(kname(v))
            out.add(v)
    return TSet(out, frozenset(sites))


def __tblc__(src, mode, fn):
    """Key-preserving comprehension over a table: the result keeps the sites."""
    if isinstance(src, TDict):
        out = fn(dict.items(src) if mode == "items" else dict.keys(src))
        o = TDict(out)
        o._sites = src._sites
        return o
    if mode == "items":
        return fn(src.items())
    if mode == "keys":
        return fn(src.keys())
    return fn(src)


def _derivable(node):
    """(source expr, mode) when the comprehension maps each source key to itself."""
    if len(node.generators) != 1:
        return None
    g = node.generators[0]
    if g.is_async or not isinstance(node.key, ast.Name):
        return None
    t, it = g.target, g.iter
    if isinstance(t, ast.Tuple):
        if (len(t.elts) != 2 or not all(isinstance(e, ast.Name) for e in t.elts)
                or t.elts[0].id != node.key.id or t.elts[1].id == node.key.id):
            return None
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "items"
                and not it.args and not it.keywords):
            return it.func.value, "items"
        return None
    if isinstance(t, ast.Name) and t.id == node.key.id:
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "keys"
                and not it.args and not it.keywords):
            return it.func.value, "keys"
        if isinstance(it, (ast.Name, ast.Attribute)):
            return it, "iter"
    return None


class _Rewrite(ast.NodeTransformer):
    def __init__(self, rel: str):
        self.rel = rel
        self.depth = 0

    def visit_FunctionDef(self, node):
        for d in node.decorator_list:
            self.visit(d)
        self.visit(node.args)
        if node.returns:
            self.visit(node.returns)
        self.depth += 1
        node.body = [self.visit(s) for s in node.body]
        self.depth -= 1
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self.visit(node.args)
        self.depth += 1
        node.body = self.visit(node.body)
        self.depth -= 1
        return node

    def visit_DictComp(self, node):
        self.generic_visit(node)
        if self.depth != 0:
            return node
        d = _derivable(node)
        if d is None or any(isinstance(n, ast.NamedExpr) for n in ast.walk(node)):
            return node
        src, mode = d
        node.generators[0].iter = ast.Name(id="__tbl_it", ctx=ast.Load())
        lam = ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="__tbl_it")], kwonlyargs=[],
                               kw_defaults=[], defaults=[]),
            body=node)
        call = ast.Call(func=ast.Name(id="__tblc__", ctx=ast.Load()),
                        args=[src, ast.Constant(mode), lam], keywords=[])
        return ast.copy_location(call, node)

    def _set_value(self, value):
        items = []
        for e in value.elts:
            star = isinstance(e, ast.Starred)
            v = e.value if star else e
            items.append(ast.Tuple(elts=[ast.Constant(e.lineno), ast.Constant(e.end_lineno),
                                         ast.Constant(star), v], ctx=ast.Load()))
        site = f"{self.rel}:{value.lineno}:{value.col_offset}"
        call = ast.Call(func=ast.Name(id="__tbls__", ctx=ast.Load()),
                        args=[ast.Constant(site), ast.List(elts=items, ctx=ast.Load())], keywords=[])
        return ast.copy_location(call, value)

    def _dict_value(self, value):
        if self.depth == 0 and isinstance(value, ast.Set) and SETS:
            return self._set_value(value)
        if self.depth != 0 or not isinstance(value, ast.Dict) or not value.keys:
            return value
        items = []
        for k, v in zip(value.keys, value.values):
            lo = (k.lineno if k is not None else v.lineno)
            hi = v.end_lineno
            items.append(ast.Tuple(elts=[
                ast.Constant(lo), ast.Constant(hi), ast.Constant(k is None),
                k if k is not None else ast.Constant(None), v,
            ], ctx=ast.Load()))
        site = f"{self.rel}:{value.lineno}:{value.col_offset}"
        call = ast.Call(func=ast.Name(id="__tbl__", ctx=ast.Load()),
                        args=[ast.Constant(site), ast.List(elts=items, ctx=ast.Load())], keywords=[])
        return ast.copy_location(call, value)

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            node.value = self._dict_value(node.value)
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value is not None and isinstance(node.target, ast.Name):
            node.value = self._dict_value(node.value)
        return node


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        src = self.get_data(path)
        rel = os.path.relpath(os.path.realpath(path), ROOT)
        tree = ast.parse(src, filename=path)
        tree = _Rewrite(rel).visit(tree)
        ast.fix_missing_locations(tree)
        FILES.append(rel)
        return compile(tree, path, "exec", dont_inherit=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "sqlglot" and not name.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            return spec
        if not os.path.realpath(spec.origin).startswith(PKG):
            return spec
        spec.loader = _Loader(spec.loader.name, spec.loader.path)
        return spec


class _SyncPool:
    """ProcessPoolExecutor stand-in that runs work in-process, through pickle."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            result = fn(*args, **kwargs)
            future.set_result(pickle.loads(pickle.dumps(result)))
        except BaseException as e:  # noqa: BLE001
            future.set_exception(e)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futures = [self.submit(fn, *a) for a in zip(*iterables)]
        return (f.result() for f in futures)

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _switch(key):
    b = BUCKETS.setdefault(key, {"look": set(), "all": set()})
    b["look"] |= LOOK
    b["all"] |= ALL
    LOOK.clear()
    ALL.clear()


import builtins  # noqa: E402

builtins.__tbl__ = __tbl__
builtins.__tblc__ = __tblc__
builtins.__tbls__ = __tbls__
if "sqlglot" in sys.modules:
    raise RuntimeError("tbl_plugin must load before sqlglot")
sys.meta_path.insert(0, _Finder())
concurrent.futures.ProcessPoolExecutor = _SyncPool
import concurrent.futures.process as _cfp  # noqa: E402

_cfp.ProcessPoolExecutor = _SyncPool

import pytest  # noqa: E402

OUTCOMES: dict = {}


def pytest_configure(config):
    import pkgutil

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if info.name.endswith("__main__"):
            continue
        try:
            __import__(info.name)
        except Exception as e:  # noqa: BLE001
            print(f"tbl: cannot import {info.name}: {e}", file=sys.stderr)
    _switch("import")
    if WHO is not None:
        json.dump({k: sorted(v) for k, v in WHO.items()}, open(os.environ["TBL_WHO"], "w"), indent=0)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    _switch("other")
    yield
    path = getattr(collector, "path", None)
    _switch(f"collect:{path}" if path is not None else "other")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    from _pytest.runner import runtestprotocol

    _switch("other")
    config = item.config
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        code = 0
        try:
            OUTCOMES.clear()
            reports = runtestprotocol(item, nextitem=None, log=False)
            payload = {
                "look": LOOK, "all": ALL, "outcomes": dict(OUTCOMES),
                "reports": [config.hook.pytest_report_to_serializable(config=config, report=r) for r in reports],
            }
            with os.fdopen(wfd, "wb") as fh:
                fh.write(pickle.dumps(payload))
        except BaseException:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            code = 1
        finally:
            os._exit(code)
    os.close(wfd)
    with os.fdopen(rfd, "rb") as fh:
        data = fh.read()
    os.waitpid(pid, 0)
    if not data:
        OUTCOMES[item.nodeid] = "failed"
        print(f"tbl: fork for {item.nodeid} sent nothing", file=sys.stderr)
    else:
        payload = pickle.loads(data)
        b = BUCKETS.setdefault(f"test:{item.nodeid}", {"look": set(), "all": set()})
        b["look"] |= payload["look"]
        b["all"] |= payload["all"]
        for nid, o in payload["outcomes"].items():
            if o == "failed":
                OUTCOMES[nid] = "failed"
            else:
                OUTCOMES.setdefault(nid, o)
        for r in payload["reports"]:
            rep = config.hook.pytest_report_from_serializable(config=config, data=r)
            item.ihook.pytest_runtest_logreport(report=rep)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def pytest_runtest_logreport(report):
    if report.failed:
        OUTCOMES[report.nodeid] = "failed"
    else:
        OUTCOMES.setdefault(report.nodeid, "ok")


def pytest_sessionfinish(session):
    _switch("other")
    out = {
        "files": sorted(set(FILES)),
        "entries": {s: [[lo, hi, sorted(n)] for (lo, hi), n in sorted(e.items())] for s, e in ENTRIES.items()},
        "buckets": {k: {"look": sorted(map(list, b["look"])), "all": sorted(b["all"])}
                    for k, b in BUCKETS.items() if b["look"] or b["all"]},
        "outcomes": OUTCOMES,
    }
    with open(os.environ["TBL_OUT"], "w") as f:
        json.dump(out, f)
