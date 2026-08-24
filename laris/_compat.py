"""Backward-compatibility shims for LARIS's public API.

Several public functions historically named their polymorphic argument
``adata``/``lr_adata`` even though it accepts an AnnData, a cytome
``Dataset``, or a path to a ``.cytome`` file. Those arguments are
standardised to ``data`` and ``lr_data``; the old keywords keep working as
deprecated aliases so existing notebooks do not break.

Mirrors ``piaso.tools._compat``. Kept as a copy rather than an import so
that LARIS installs without piaso-tools; keep the two in step if the
convention changes.
"""

import warnings

# Sentinel distinguishing "argument not supplied" from an explicit ``None``.
_UNSET = object()


def resolve_data_arg(data, func_name, canonical='data', required=True,
                     **aliases):
    """Resolve a polymorphic argument from its new name and legacy aliases.

    Parameters
    ----------
    data : object
        Value of the new parameter (``_UNSET`` if not supplied).
    func_name : str
        Name of the calling function, used in the messages.
    canonical : str, default='data'
        The new parameter's name, quoted in messages. LARIS has two such
        slots: ``data`` (expression) and ``lr_data`` (LR scores).
    required : bool, default=True
        When False, an absent argument resolves to ``None`` instead of
        raising - for genuinely optional arguments such as ``runLARIS``'s
        expression object, which is only needed when ``by_celltype=True``.
    **aliases :
        Legacy keyword aliases (e.g. ``adata=...``), each ``_UNSET`` if not
        supplied. Passing one emits a ``FutureWarning`` and is used as the
        argument's value.

    Returns
    -------
    The resolved object (AnnData, cytome ``Dataset``, or path string).
    """
    given = [(name, val) for name, val in aliases.items() if val is not _UNSET]
    data_given = data is not _UNSET
    if given:
        if data_given or len(given) > 1:
            names = ", ".join(
                ([canonical] if data_given else []) + [n for n, _ in given]
            )
            raise TypeError(
                f"{func_name}() received more than one of ({names}); pass "
                f"only `{canonical}`."
            )
        name, val = given[0]
        warnings.warn(
            f"`{name}=` is deprecated in {func_name}(); use `{canonical}=` "
            f"(it accepts an AnnData, a cytome Dataset, or a path to a "
            f".cytome file).",
            FutureWarning, stacklevel=3,
        )
        return val
    if not data_given:
        if not required:
            return None
        raise TypeError(
            f"{func_name}() missing required argument: '{canonical}'"
        )
    return data
