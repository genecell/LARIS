"""The API reference in ``api/`` is generated from the docstrings, so it
cannot drift from the code:

1. the committed pages equal what ``tools/make_api_docs.py`` generates now;
2. numpydoc's structural checks pass on every public object: every
   parameter documented (PR01), none unknown (PR02), in signature order
   (PR03), a Returns section (RT01), no unknown section (GL06), sections in
   the standard order (GL07). Cosmetic checks (final periods, summaries on
   one line, See Also on everything) are not enforced.
"""
import importlib.util
import inspect
import warnings
from pathlib import Path

import pytest

pytest.importorskip("numpydoc")

ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL = {"PR01", "PR02", "PR03", "RT01", "GL06", "GL07"}
# Result containers built by LARIS and never constructed by users document
# their fields as Attributes, not as constructor parameters.
ALLOWED = {("laris.tl.LRBackground", "PR01")}


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_api_docs", ROOT / "tools" / "make_api_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _public():
    import laris as la
    for mod in ("tl", "pl", "pp", "datasets"):
        m = getattr(la, mod)
        names = getattr(m, "__all__", None) or [
            n for n in dir(m) if not n.startswith("_")]
        for n in names:
            obj = getattr(m, n, None)
            if inspect.isfunction(obj) or inspect.isclass(obj):
                yield f"laris.{mod}.{n}"


def test_api_pages_are_current():
    pages = _generator().build_pages()
    stale = [n for n, text in pages.items()
             if not (ROOT / "api" / n).exists()
             or (ROOT / "api" / n).read_text() != text]
    assert not stale, (f"api/ is stale ({', '.join(stale)}): run "
                       f"python tools/make_api_docs.py and commit the result")


def test_every_public_function_has_a_section():
    pages = _generator().build_pages()
    for name in _public():
        _, mod, fn = name.split(".")
        page = pages[f"{mod}.md"]
        # its own section, or a row pointing to the page that documents it
        # (functions exported by two modules, e.g. readCytome)
        assert (f"`la.{mod}.{fn}`" in page
                or f"| `{fn}` | Same function as" in page), (
            f"{name} missing from api/{mod}.md")


@pytest.mark.parametrize("name", sorted(_public()))
def test_docstring_structure(name):
    from numpydoc.validate import validate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = validate(name)["errors"]
    bad = [f"{code} {msg}" for code, msg in errors
           if code in STRUCTURAL and (name, code) not in ALLOWED]
    assert not bad, f"{name}:\n" + "\n".join(bad)
