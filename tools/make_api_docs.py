"""Generate LARIS's API reference (``api/*.md``) from its docstrings.

The pages are plain GitHub Markdown, like the tutorials: readable on GitHub,
in an editor or by a coding agent, with no website build. They are derived
entirely from the NumPy-style docstrings, parsed with numpydoc (the
reference parser for that format; a development-only dependency), so the
docstrings stay the single source.

    python tools/make_api_docs.py           # rewrite api/*.md
    python tools/make_api_docs.py --check   # exit 1 if any page is stale

``tests/test_api_docs.py`` runs the same check, plus numpydoc's structural
validation of every public object.
"""
import argparse
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "api"

MODULES = [
    ("tl", "Analysis",
     "LR scores, the matched-gene background and exact P values, "
     "cross-condition comparison, decoy controls."),
    ("pl", "Plotting", "Heatmaps, networks, dot plots, spatial maps and comparison volcanoes."),
    ("pp", "Reading data", "Cytome readers and multi-sample coordinate offsets."),
    ("datasets", "Datasets", "The bundled ligand-receptor database."),
]
# deprecated aliases: documented in the tables, left out of the signatures
DEPRECATED = {"adata", "lr_adata", "results"}
# public objects that are not functions or classes
SKIP = {("pl", "pos_cmap")}

_ROLE = re.compile(r":(?:func|class|meth|attr|mod|data|obj|exc):`~?([^`]+)`")
_BULLET = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


def _inline(text: str) -> str:
    """reST inline markup to Markdown: roles and ``literals`` to code."""
    return _ROLE.sub(r"`\1`", text).replace("``", "`")


def _dedent(lines):
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return []
    pad = min(len(ln) - len(ln.lstrip()) for ln in body)
    return [ln[pad:] if ln.strip() else "" for ln in lines]


def _blocks(lines):
    """Markdown for a free-text section. Paragraphs and lists pass through;
    doctest blocks, '::' literal blocks and indented blocks (formulas)
    become fenced code."""
    lines = _dedent(list(lines))
    out, i, n = [], 0, len(lines)
    prev_text = ""                       # last non-blank line emitted as text
    while i < n:
        ln = lines[i]
        if ln.lstrip().startswith(">>>"):
            j = i
            while j < n and lines[j].strip():
                j += 1
            out += ["```python"] + _dedent(lines[i:j]) + ["```"]
            i = j
            continue
        if ln.rstrip().endswith("::"):
            head = ln.rstrip()[:-2].rstrip()
            if head:
                out.append(_inline(head) + ":")
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            k = j
            while k < n and (not lines[k].strip() or lines[k].startswith(" ")):
                k += 1
            code = _dedent(lines[j:k])
            while code and not code[-1].strip():
                code.pop()
            out += ["", "```text"] + code + ["```", ""]
            i = k
            continue
        indent = len(ln) - len(ln.lstrip())
        after_blank = i > 0 and not lines[i - 1].strip()
        if (ln.strip() and indent >= 4 and after_blank
                and not _BULLET.match(prev_text)):
            k = i
            while k < n and (not lines[k].strip() or
                             len(lines[k]) - len(lines[k].lstrip()) >= 4):
                k += 1
            code = _dedent(lines[i:k])
            while code and not code[-1].strip():
                code.pop()
            out += ["```text"] + code + ["```", ""]
            i = k
            continue
        out.append(_inline(ln))
        if ln.strip():
            prev_text = ln
        i += 1
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _cell(lines) -> str:
    """One table cell: paragraphs joined by <br><br>, list items by <br>."""
    paras, cur = [], []
    for ln in _dedent(list(lines)):
        if not ln.strip():
            if cur:
                paras.append(cur)
                cur = []
            continue
        cur.append(ln)
    if cur:
        paras.append(cur)
    out = []
    for p in paras:
        items, buf = [], []
        for ln in p:
            if _BULLET.match(ln):
                if buf:
                    items.append(" ".join(buf))
                buf = ["• " + _BULLET.sub("", ln).strip()]
            else:
                buf.append(ln.strip())
        if buf:
            items.append(" ".join(buf))
        out.append("<br>".join(items))
    return _inline("<br><br>".join(out)).replace("|", "\\|")


def _documented_defaults(obj) -> dict:
    """{parameter: default} as written in the docstring ("default=...")."""
    d = _doc(obj)
    out = {}
    for sec in ("Parameters", "Other Parameters"):
        for p in d[sec]:
            m = re.search(r"default\s*=\s*(.+)$", p.type)
            if m:
                for name in p.name.split(","):
                    out[name.strip()] = m.group(1).strip()
    return out


def _signature(qual, obj) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    documented = _documented_defaults(obj)
    parts = []
    for p in sig.parameters.values():
        if p.name in DEPRECATED:
            continue
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
            continue
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
            continue
        d = p.default
        if d is inspect.Parameter.empty:
            parts.append(p.name)
        elif " at 0x" in repr(d):
            # a "not passed" sentinel: show the documented default, else the
            # parameter alone (the data arguments; the table says if optional)
            parts.append(f"{p.name}={documented[p.name]}" if p.name in documented
                         else p.name)
        else:
            parts.append(f"{p.name}={d!r}")
    if inspect.isclass(obj):
        return ""
    body = ",\n    ".join(parts)
    return f"```python\n{qual}(\n    {body},\n)\n```" if parts else f"```python\n{qual}()\n```"


def _param_table(items) -> list:
    rows = ["| Parameter | Type | Description |", "|---|---|---|"]
    for p in items:
        names = ", ".join(f"`{n.strip()}`" for n in p.name.split(","))
        rows.append(f"| {names} | {_inline(p.type).replace('|', chr(92) + '|')} | {_cell(p.desc)} |")
    return rows


def _returns(items) -> list:
    out = []
    for p in items:
        head = f"`{p.name}` : {_inline(p.type)}" if p.name else _inline(p.type)
        desc = _blocks(p.desc)
        out.append(f"- **{head}**")
        if desc:
            out += ["", "\n".join("  " + ln if ln else "" for ln in desc.split("\n")), ""]
    return out


def _anchor(name: str) -> str:
    return name.lower()


def _doc(obj):
    from numpydoc.docscrape import ClassDoc, FunctionDoc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ClassDoc(obj) if inspect.isclass(obj) else FunctionDoc(obj)


def _section(mod: str, name: str, obj) -> str:
    qual = f"la.{mod}.{name}"
    d = _doc(obj)
    lines = [f'<a id="{_anchor(name)}"></a>', "", f"### `{qual}`", ""]
    sig = _signature(qual, obj)
    if sig:
        lines += [sig, ""]
    if d["Summary"]:
        lines += [_inline(" ".join(s.strip() for s in d["Summary"])), ""]
    if d["Extended Summary"]:
        lines += [_blocks(d["Extended Summary"]), ""]
    for sec in ("Parameters", "Other Parameters", "Attributes"):
        if d[sec]:
            lines += [f"**{sec}**", ""] + _param_table(d[sec]) + [""]
    for sec in ("Returns", "Yields", "Raises", "Warns"):
        if d[sec]:
            lines += [f"**{sec}**", ""] + _returns(d[sec]) + [""]
    for sec in ("Warnings", "Notes", "References"):
        if d[sec]:
            lines += [f"**{sec}**", "", _blocks(d[sec]), ""]
    if d["See Also"]:
        refs = []
        for names, _ in d["See Also"]:
            refs += [f"`{n}`" for n, _role in names]
        lines += ["**See also:** " + ", ".join(refs), ""]
    if d["Examples"]:
        lines += ["**Examples**", "", _blocks(d["Examples"]), ""]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def _members(mod: str):
    import laris
    m = getattr(laris, mod)
    names = getattr(m, "__all__", None) or [n for n in dir(m) if not n.startswith("_")]
    out = []
    for n in names:
        if (mod, n) in SKIP:
            continue
        obj = getattr(m, n, None)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            out.append((n, obj))
    return out


def _owner(obj):
    """The module page that documents an object exported by several: the
    one matching the package that defines it."""
    home = {"preprocessing": "pp", "plotting": "pl", "tools": "tl",
            "datasets": "datasets"}
    parts = inspect.getmodule(obj).__name__.split(".")
    return home.get(parts[1]) if len(parts) > 1 else None


def _summary(obj) -> str:
    d = _doc(obj)
    return _inline(" ".join(s.strip() for s in d["Summary"])).replace("|", "\\|")


def build_pages() -> dict:
    """{file name: Markdown} for every page of the API reference."""
    pages = {}
    index = ["# LARIS API reference", "",
             "<!-- Generated from the docstrings by tools/make_api_docs.py; do not edit by hand. -->", "",
             "Every public function, generated from its docstring. For worked "
             "examples on real data see the [tutorials](../tutorials/README.md).", ""]
    for mod, title, blurb in MODULES:
        members = _members(mod)
        page = [f"# `laris.{mod}`: {title}", "",
                "<!-- Generated from the docstrings by tools/make_api_docs.py; do not edit by hand. -->", "",
                blurb, "", "| Function | Summary |", "|---|---|"]
        index += [f"## [`laris.{mod}`]({mod}.md): {title}", "", blurb, "",
                  "| Function | Summary |", "|---|---|"]
        body = []
        for name, obj in members:
            owner = _owner(obj)
            if owner is not None and owner != mod:
                line = f"| `{name}` | Same function as [`la.{owner}.{name}`]({owner}.md#{_anchor(name)}). |"
                page.append(line)
                index.append(line.replace(f"({owner}.md#", f"({owner}.md#"))
                continue
            summary = _summary(obj)
            page.append(f"| [`{name}`](#{_anchor(name)}) | {summary} |")
            index.append(f"| [`{name}`]({mod}.md#{_anchor(name)}) | {summary} |")
            body.append(_section(mod, name, obj))
        page += ["", "---", ""] + ["\n---\n\n".join(body)]
        pages[f"{mod}.md"] = re.sub(r"\n{3,}", "\n\n", "\n".join(page)).rstrip() + "\n"
        index.append("")
    pages["README.md"] = "\n".join(index).rstrip() + "\n"
    return pages


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a committed page differs from the docstrings")
    args = ap.parse_args(argv)
    pages = build_pages()
    stale = [n for n, text in pages.items()
             if not (API / n).exists() or (API / n).read_text() != text]
    if args.check:
        for n in stale:
            print(f"api/{n} is stale: run python tools/make_api_docs.py")
        return 1 if stale else 0
    API.mkdir(exist_ok=True)
    for n, text in pages.items():
        (API / n).write_text(text)
    print(f"wrote {len(pages)} pages to {API} ({len(stale)} changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
