//! Optional acceleration kernels for LARIS's factorized null.
//!
//! Kernel 1 computes the masked co-localization numerator via the sparse
//! decomposition Pnz = P - Q (support = raw-nonzero cells):
//!
//!     N = (P.*y) @ Pnz' + (Pnz.*y) @ P' - (Pnz.*y) @ Pnz'
//!
//! With A[i,j] = sum_c P[i,c] y[c] Pnz[j,c] the three terms collapse to
//! N = A + A' - B (B = (Pnz.*y) @ Pnz', symmetric). A is built one output
//! column at a time as contiguous axpy over P's columns (P passed
//! column-major), B via sparse-column scatter; both rayon-parallel over
//! output columns, which write disjoint memory. Flop count is
//! density * (dense cost); the dense BLAS route wins on raw efficiency
//! but loses the density factor - crossover measured per dataset.
//!
//! Kernel 2 fuses the per-row null assembly (gather + multiply + count).

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ctc_cos_numerator<'py>(
    py: Python<'py>,
    pt: PyReadonlyArray2<'py, f32>,          // P transposed: (n_cells, U), C-order
    csr_indptr: PyReadonlyArray1<'py, i64>,  // Pnz CSR (U rows)
    csr_indices: PyReadonlyArray1<'py, i32>,
    csr_data: PyReadonlyArray1<'py, f32>,
    csc_indptr: PyReadonlyArray1<'py, i64>,  // Pnz CSC (n_cells cols)
    csc_indices: PyReadonlyArray1<'py, i32>,
    csc_data: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let pt = pt.as_array();
    let n = pt.shape()[0];
    let u = pt.shape()[1];
    let pt_flat = pt.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("pt must be C-contiguous")
    })?;
    let rp = csr_indptr.as_slice()?;
    let ri = csr_indices.as_slice()?;
    let rd = csr_data.as_slice()?;
    let cp = csc_indptr.as_slice()?;
    let ci = csc_indices.as_slice()?;
    let cd = csc_data.as_slice()?;
    let y = y.as_slice()?;
    debug_assert_eq!(cp.len(), n + 1);

    // one parallel pass builds A and B per output column (disjoint writes)
    let mut a_mat = vec![0f32; u * u];       // column-major
    let mut b_mat = vec![0f32; u * u];
    py.allow_threads(|| {
        a_mat.par_chunks_mut(u).zip(b_mat.par_chunks_mut(u)).enumerate()
            .for_each(|(j, (acol, bcol))| {
                let start = rp[j] as usize;
                let stop = rp[j + 1] as usize;
                for e in start..stop {
                    let c = ri[e] as usize;
                    let w = y[c] * rd[e];
                    let pcol = &pt_flat[c * u..(c + 1) * u];
                    for (a, p) in acol.iter_mut().zip(pcol.iter()) {
                        *a += w * p;
                    }
                    let cs = cp[c] as usize;
                    let ce = cp[c + 1] as usize;
                    for f in cs..ce {
                        bcol[ci[f] as usize] += w * cd[f];
                    }
                }
            });
    });
    // N = A + A' - B  (B symmetric; a_mat/b_mat column-major)
    let mut out = vec![0f32; u * u];
    py.allow_threads(|| {
        out.par_chunks_mut(u).enumerate().for_each(|(i, row)| {
            for j in 0..u {
                row[j] = a_mat[j * u + i] + a_mat[i * u + j] - b_mat[j * u + i];
            }
        });
    });
    let arr = Array2::from_shape_vec((u, u), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

/// Fused null assembly: per tested row, gather + multiply + count.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn assembly_counts<'py>(
    py: Python<'py>,
    m_flat: PyReadonlyArray1<'py, f64>,
    spec_s: PyReadonlyArray1<'py, f64>,
    spec_r: PyReadonlyArray1<'py, f64>,
    flat: PyReadonlyArray1<'py, i64>,
    row_gene: PyReadonlyArray1<'py, i32>,
    col_gene: PyReadonlyArray1<'py, i32>,
    dw: PyReadonlyArray1<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    pair_of: PyReadonlyArray1<'py, i64>,
    obs: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let m = m_flat.as_slice()?;
    let ss = spec_s.as_slice()?;
    let sr = spec_r.as_slice()?;
    let flat = flat.as_slice()?;
    let rg = row_gene.as_slice()?;
    let cg = col_gene.as_slice()?;
    let dw = dw.as_slice()?;
    let off = offsets.as_slice()?;
    let pair_of = pair_of.as_slice()?;
    let obs = obs.as_slice()?;
    let nrows = obs.len();
    let mut exceed = vec![0i64; nrows];
    // count of pseudo-pairs with a POSITIVE score: the null's effective
    // support. Zero-valued entries can never exceed the observed score,
    // so they add to the denominator without adding resolution.
    let mut npos = vec![0i64; nrows];
    py.allow_threads(|| {
        exceed.par_iter_mut().zip(npos.par_iter_mut()).enumerate()
            .for_each(|(r, (ex, ap))| {
                let pid = pair_of[r] as usize;
                let s = off[pid] as usize;
                let e = off[pid + 1] as usize;
                let o = obs[r];
                let mut cnt = 0i64;
                let mut pos = 0i64;
                for t in s..e {
                    let v = ss[rg[t] as usize] * sr[cg[t] as usize]
                        * dw[t] * m[flat[t] as usize];
                    if v > 0.0 { pos += 1; }
                    if v >= o { cnt += 1; }
                }
                *ex = cnt;
                *ap = pos;
            });
    });
    Ok((Array1::from(exceed).into_pyarray_bound(py),
        Array1::from(npos).into_pyarray_bound(py)))
}

#[pymodule]
fn _laris(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ctc_cos_numerator, m)?)?;
    m.add_function(wrap_pyfunction!(assembly_counts, m)?)?;
    Ok(())
}
