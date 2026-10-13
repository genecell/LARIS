# `laris.datasets`: Datasets

<!-- Generated from the docstrings by tools/make_api_docs.py; do not edit by hand. -->

The bundled ligand-receptor database.

| Function | Summary |
|---|---|
| [`lrDatabase`](#lrdatabase) | Load a bundled ligand-receptor interaction database. |

---

<a id="lrdatabase"></a>

### `la.datasets.lrDatabase`

```python
la.datasets.lrDatabase(
    species='human',
    pathway=None,
    annotation=None,
)
```

Load a bundled ligand-receptor interaction database.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `species` | str, default 'human' | Species for the database. One of 'human' or 'mouse'. |
| `pathway` | str, optional | Filter to a specific signaling pathway (e.g., 'WNT', 'TGFb'). Matches the `pathway_name` column. Case-sensitive. |
| `annotation` | str, optional | Filter to a specific annotation category (e.g., 'Secreted Signaling', 'ECM-Receptor', 'Cell-Cell Contact'). Matches the `annotation` column. Case-sensitive. |

**Returns**

- **pd.DataFrame**

  Ligand-receptor database with columns including `interaction_name`,
  `pathway_name`, `ligand`, `receptor`, `annotation`, etc.

**Raises**

- **ValueError**

  If `species` is not 'human' or 'mouse'.

**Examples**

```python
>>> import laris as la
>>> lr_human = la.datasets.lrDatabase('human')
>>> lr_mouse = la.datasets.lrDatabase('mouse')
>>> lr_wnt = la.datasets.lrDatabase('human', pathway='WNT')
```
