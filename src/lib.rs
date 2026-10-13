//! Optional acceleration kernels for LARIS's factorized null.
//!
//! Kernel 1 computes the masked co-localization numerator via the sparse
//! decomposition Pnz = P - Q (support = raw-nonzero cells):
//!
//!     N = (P.*y) @ Pnz' + (Pnz.*y) @ P' - (Pnz.*y) @ Pnz'
//!
//! With A[i,j] = sum_c P[i,c] y[c] Pnz[j,c] the three terms collapse to
//! N = A + A' - B (B = (Pnz.*y) @ Pnz', symmetric). Each entry of A is an
//! axpy contribution of a P row segment per detected (gene, cell), B a
//! sparse scatter; both are accumulated over cells in increasing order.
//! Flop count is density * (dense cost). Since v0.15.0 the work is
//! blocked (ROW_TILE rows x GENE_BLOCK columns per task) so that each P
//! row segment is read once for every column of the block it feeds,
//! instead of once per (column, cell); the per-entry order is unchanged,
//! so A and B are bitwise those of the v0.14 column-at-a-time kernel
//! (tonsil: 3.5x faster with dense co-location, 12x when sparse). A and B
//! go into buffers the caller reuses, and the kernels below read N from
//! them, so N itself is never stored.
//!
//! The remaining kernels replace the NumPy element-wise chain of the
//! P-value step (v0.15.0). Each one performs, per table entry, exactly the
//! IEEE operations the NumPy code performs, in the same order (Rust never
//! contracts a*b+c into an FMA), so the tables, counts and P values are
//! bitwise identical to the NumPy expressions they replace:
//!
//! * `cos_sq_accumulate`: pass A, S += c*c (and c.T*c.T for the mirror
//!   cell-type pair), c = N / (||v|| ||y||);
//! * `coloc_x`: pass B up to the IQR division, lam * c / iqr, left for
//!   NumPy's log1p (whose SIMD implementation is not libm's);
//! * `assembly_sweep`: M = coloc * frac_s * frac_r * scale one row at a
//!   time, and the exceedance and support counts of every tested row of a
//!   cell-type pair and its mirror in one sweep over the table rows.

use std::sync::atomic::{AtomicI64, Ordering};

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
            PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// A raw pointer that rayon tasks may share. Every task writes a disjoint
/// (row tile, column block) region of the matrix behind it.
#[derive(Clone, Copy)]
struct SharedMut(*mut f32);
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}

/// Columns of A and B per task, and the granularity of the per-cell
/// pointers into the CSC columns.
const GENE_BLOCK: usize = 256;
/// Rows of A and B per task; a multiple of GENE_BLOCK. A task's A region
/// (ROW_TILE x GENE_BLOCK floats, 512 KB) stays in L2 while every cell's
/// P segment is read once for all ~GENE_BLOCK * density columns it feeds.
const ROW_TILE: usize = 512;
// A task's B updates cover whole gene blocks of its rows: a ROW_TILE that
// is not a multiple of GENE_BLOCK would write outside the task's region.
const _: () = assert!(ROW_TILE % GENE_BLOCK == 0);

/// One task of the blocked numerator: A[i0..i1, cols of block jb] and the
/// matching B entries, accumulated over cells in increasing order - the
/// order in which the column-at-a-time kernel adds them, so every entry of
/// A and B is bitwise the same.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn numerator_task(t: usize, jb: usize, u: usize, n: usize, nb: usize,
                  pt: &[f32], bptr: &[usize], ci: &[i32], cd: &[f32],
                  y: &[f32], a: SharedMut, b: SharedMut) {
    let i0 = t * ROW_TILE;
    let i1 = (i0 + ROW_TILE).min(u);
    let (g0, g1) = (i0 / GENE_BLOCK, i1.div_ceil(GENE_BLOCK));
    for c in 0..n {
        let yc = y[c];
        // exactly-zero co-location: skipping adds nothing (see above)
        if yc == 0.0 {
            continue;
        }
        let bp = &bptr[c * (nb + 1)..(c + 1) * (nb + 1)];
        let (e0, e1) = (bp[jb], bp[jb + 1]);
        if e0 == e1 {
            continue;
        }
        let seg = &pt[c * u + i0..c * u + i1];
        let (f0, f1) = (bp[g0], bp[g1]);
        for e in e0..e1 {
            let j = ci[e] as usize;
            let w = yc * cd[e];
            // SAFETY: (rows i0..i1, column j) belongs to this task alone
            let acol = unsafe {
                std::slice::from_raw_parts_mut(a.0.add(j * u + i0), i1 - i0)
            };
            for (av, p) in acol.iter_mut().zip(seg.iter()) {
                *av += w * p;
            }
            let bcol = unsafe { b.0.add(j * u) };
            for f in f0..f1 {
                unsafe { *bcol.add(ci[f] as usize) += w * cd[f]; }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn numerator_task_avx2(t: usize, jb: usize, u: usize, n: usize, nb: usize,
                              pt: &[f32], bptr: &[usize], ci: &[i32], cd: &[f32],
                              y: &[f32], a: SharedMut, b: SharedMut) {
    numerator_task(t, jb, u, n, nb, pt, bptr, ci, cd, y, a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn numerator_task_avx512(t: usize, jb: usize, u: usize, n: usize, nb: usize,
                                pt: &[f32], bptr: &[usize], ci: &[i32], cd: &[f32],
                                y: &[f32], a: SharedMut, b: SharedMut) {
    numerator_task(t, jb, u, n, nb, pt, bptr, ci, cd, y, a, b)
}

/// N[i, j] = A[i, j] + A[j, i] - B[i, j] from the column-major A and B of
/// the numerator (the one combination formula, used everywhere).
#[inline(always)]
fn n_at(a: &[f32], b: &[f32], u: usize, i: usize, j: usize) -> f32 {
    a[j * u + i] + a[i * u + j] - b[j * u + i]
}

/// A and B of the masked co-localization numerator N = A + A' - B (see the
/// module notes), written into `a` and `b` (U x U, column-major), blocked
/// for cache reuse: tasks own (ROW_TILE rows, GENE_BLOCK columns) of A and
/// B and sweep the cells once. Wider SIMD is chosen at run time; it only
/// changes how many independent entries one instruction updates (no FMA is
/// ever formed), so the result is bitwise the same on every CPU.
#[allow(clippy::too_many_arguments)]
fn numerator_ab(py: Python<'_>, pt: &PyReadonlyArray2<'_, f32>,
                csc_indptr: &PyReadonlyArray1<'_, i64>,
                csc_indices: &PyReadonlyArray1<'_, i32>,
                csc_data: &PyReadonlyArray1<'_, f32>,
                y: &PyReadonlyArray1<'_, f32>,
                a: &mut [f32], b: &mut [f32]) -> PyResult<()> {
    let n = pt.shape()[0];
    let u = pt.shape()[1];
    if !pt.is_c_contiguous() {
        return Err(PyValueError::new_err("pt must be C-contiguous"));
    }
    let pt_flat = pt.as_slice()?;
    let cp = csc_indptr.as_slice()?;
    let ci = csc_indices.as_slice()?;
    let cd = csc_data.as_slice()?;
    let y = y.as_slice()?;
    if cp.len() != n + 1 || y.len() != n || ci.len() != cd.len()
        || cp[n] as usize != ci.len() || a.len() != u * u || b.len() != u * u {
        return Err(PyValueError::new_err("numerator: inconsistent sizes"));
    }
    // per cell, the first CSC entry of every gene block (genes ascending)
    let nb = u.div_ceil(GENE_BLOCK);
    let mut bptr = vec![0usize; n * (nb + 1)];
    let sorted = py.allow_threads(|| {
        bptr.par_chunks_mut(nb + 1).enumerate().map(|(c, bp)| {
            let (s, e) = (cp[c] as usize, cp[c + 1] as usize);
            let col = &ci[s..e];
            if col.windows(2).any(|w| w[0] >= w[1])
                || col.iter().any(|&g| g < 0 || g as usize >= u) {
                return false;
            }
            let mut f = s;
            for (blk, slot) in bp.iter_mut().enumerate() {
                let lo = blk * GENE_BLOCK;
                while f < e && (ci[f] as usize) < lo {
                    f += 1;
                }
                *slot = f;
            }
            true
        }).all(|ok| ok)
    });
    if !sorted {
        return Err(PyValueError::new_err(
            "numerator: CSC gene indices must be sorted and in range"));
    }
    py.allow_threads(|| {
        a.par_chunks_mut(1 << 16).for_each(|s| s.fill(0.0));
        b.par_chunks_mut(1 << 16).for_each(|s| s.fill(0.0));
    });
    let (ap, bp) = (SharedMut(a.as_mut_ptr()), SharedMut(b.as_mut_ptr()));
    let tasks: Vec<(usize, usize)> = (0..u.div_ceil(ROW_TILE))
        .flat_map(|t| (0..nb).map(move |jb| (t, jb))).collect();
    #[cfg(target_arch = "x86_64")]
    let level = if is_x86_feature_detected!("avx512f") { 2 }
                else if is_x86_feature_detected!("avx2") { 1 } else { 0 };
    #[cfg(not(target_arch = "x86_64"))]
    let level = 0;
    let bptr = bptr.as_slice();
    py.allow_threads(|| {
        tasks.par_iter().for_each(|&(t, jb)| match level {
            // SAFETY: the CPU supports the feature (checked above)
            #[cfg(target_arch = "x86_64")]
            2 => unsafe {
                numerator_task_avx512(t, jb, u, n, nb, pt_flat, bptr, ci, cd, y, ap, bp)
            },
            #[cfg(target_arch = "x86_64")]
            1 => unsafe {
                numerator_task_avx2(t, jb, u, n, nb, pt_flat, bptr, ci, cd, y, ap, bp)
            },
            _ => numerator_task(t, jb, u, n, nb, pt_flat, bptr, ci, cd, y, ap, bp),
        });
    });
    Ok(())
}

fn writable<'a, T: numpy::Element>(
    a: &'a mut PyReadwriteArray2<'_, T>, what: &str, u: usize,
) -> PyResult<&'a mut [T]> {
    if a.shape() != [u, u] || !a.is_c_contiguous() {
        return Err(PyValueError::new_err(
            format!("{what} must be a C-contiguous ({u}, {u}) array")));
    }
    a.as_slice_mut().map_err(|_| PyValueError::new_err(
        format!("{what} must be C-contiguous")))
}

/// Fill the reusable buffers `a_buf`, `b_buf` (U x U float32) with the
/// numerator's A and B for co-location profile `y`. The P-value kernels
/// below read N from them directly, so N is never materialized.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ctc_numerator_ab<'py>(
    py: Python<'py>,
    pt: PyReadonlyArray2<'py, f32>,          // P: (n_cells, U), C-order
    csc_indptr: PyReadonlyArray1<'py, i64>,  // Pnz = P - Q as CSC (genes x
    csc_indices: PyReadonlyArray1<'py, i32>, // cells): per cell, its
    csc_data: PyReadonlyArray1<'py, f32>,    // detected genes, ascending
    y: PyReadonlyArray1<'py, f32>,
    mut a_buf: PyReadwriteArray2<'py, f32>,
    mut b_buf: PyReadwriteArray2<'py, f32>,
) -> PyResult<()> {
    let u = pt.shape()[1];
    let a = writable(&mut a_buf, "a_buf", u)?;
    let b = writable(&mut b_buf, "b_buf", u)?;
    numerator_ab(py, &pt, &csc_indptr, &csc_indices, &csc_data, &y, a, b)
}

/// The numerator N itself (U x U float32), for tests and diagnostics.
#[pyfunction]
fn ctc_cos_numerator<'py>(
    py: Python<'py>,
    pt: PyReadonlyArray2<'py, f32>,
    csc_indptr: PyReadonlyArray1<'py, i64>,
    csc_indices: PyReadonlyArray1<'py, i32>,
    csc_data: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let u = pt.shape()[1];
    let mut a = vec![0f32; u * u];
    let mut b = vec![0f32; u * u];
    numerator_ab(py, &pt, &csc_indptr, &csc_indices, &csc_data, &y, &mut a, &mut b)?;
    let mut out = vec![0f32; u * u];
    py.allow_threads(|| {
        out.par_chunks_mut(u).enumerate().for_each(|(i, row)| {
            for (j, o) in row.iter_mut().enumerate() {
                *o = n_at(&a, &b, u, i, j);
            }
        });
    });
    let arr = Array2::from_shape_vec((u, u), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

fn contiguous<'a, T: numpy::Element>(
    a: &'a PyReadonlyArray2<'_, T>, what: &str, u: usize,
) -> PyResult<&'a [T]> {
    if a.shape() != [u, u] {
        return Err(PyValueError::new_err(format!("{what} must be ({u}, {u})")));
    }
    // as_slice also accepts Fortran order, which would be read transposed
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err(format!("{what} must be C-contiguous")));
    }
    a.as_slice().map_err(|_| PyValueError::new_err(
        format!("{what} must be C-contiguous")))
}

/// cos[i, j] = N[i, j] / (||v_ij|| ||y||) where the denominator is
/// positive, else 0: the arithmetic of `_ctc_cos` (float32 N promoted to
/// float64), with N read from the numerator's A and B.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cos_ab(a: &[f32], b: &[f32], v: &[f64], y_norm: f64,
          u: usize, i: usize, j: usize) -> f64 {
    let den = v[i * u + j] * y_norm;
    if den > 0.0 { n_at(a, b, u, i, j) as f64 / den } else { 0.0 }
}

/// Rows per block and columns per tile of the element-wise kernels: the
/// transposed reads (A[j, i], B[j, i] for 16 consecutive i) are one cache
/// line per j, reused across the block's rows while the tile is in L1.
const ROWS: usize = 16;
const TILE: usize = 256;

/// Pass A: S += c*c, then S += c.T*c.T when `mirror` (the reverse
/// cell-type pair shares the co-location vector, so its cosine table is
/// this one transposed). Per entry, the two additions happen in the order
/// the NumPy statements make them, so S is bitwise identical.
#[pyfunction]
fn cos_sq_accumulate<'py>(
    py: Python<'py>,
    a_buf: PyReadonlyArray2<'py, f32>,
    b_buf: PyReadonlyArray2<'py, f32>,
    v_norm: PyReadonlyArray2<'py, f64>,
    y_norm: f64,
    mut s: PyReadwriteArray2<'py, f64>,
    mirror: bool,
) -> PyResult<()> {
    let u = a_buf.shape()[0];
    let a = contiguous(&a_buf, "a_buf", u)?;
    let b = contiguous(&b_buf, "b_buf", u)?;
    let v = contiguous(&v_norm, "v_norm", u)?;
    let s = writable(&mut s, "s", u)?;
    py.allow_threads(|| {
        s.par_chunks_mut(u * ROWS).enumerate().for_each(|(blk, sblk)| {
            let i0 = blk * ROWS;
            let rows = sblk.len() / u;
            let mut j0 = 0;
            while j0 < u {
                let j1 = (j0 + TILE).min(u);
                for di in 0..rows {
                    let i = i0 + di;
                    let srow = &mut sblk[di * u..(di + 1) * u];
                    for j in j0..j1 {
                        let c = cos_ab(a, b, v, y_norm, u, i, j);
                        let mut acc = srow[j] + c * c;
                        if mirror {
                            let ct = cos_ab(a, b, v, y_norm, u, j, i);
                            acc += ct * ct;
                        }
                        srow[j] = acc;
                    }
                }
                j0 = j1;
            }
        });
    });
    Ok(())
}

/// Pass B up to the IQR division, into `out`: x = lam * c / iqr with
/// lam = c^2 / ((1 - mu) c^2 + mu S) (0 where that is 0). The caller
/// applies NumPy's log1p to x in place: NumPy's SIMD log1p differs from
/// libm's in the last bit, and the tables must stay identical.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn coloc_x<'py>(
    py: Python<'py>,
    a_buf: PyReadonlyArray2<'py, f32>,
    b_buf: PyReadonlyArray2<'py, f32>,
    v_norm: PyReadonlyArray2<'py, f64>,
    y_norm: f64,
    s: PyReadonlyArray2<'py, f64>,
    lam_mu: f64,
    iqr: f64,
    mut out: PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let u = a_buf.shape()[0];
    let a = contiguous(&a_buf, "a_buf", u)?;
    let b = contiguous(&b_buf, "b_buf", u)?;
    let v = contiguous(&v_norm, "v_norm", u)?;
    let s = contiguous(&s, "s", u)?;
    let out = writable(&mut out, "out", u)?;
    // Python evaluates (1.0 - lam_mu) once, in double precision
    let one_minus = 1.0 - lam_mu;
    py.allow_threads(|| {
        out.par_chunks_mut(u * ROWS).enumerate().for_each(|(blk, oblk)| {
            let i0 = blk * ROWS;
            let rows = oblk.len() / u;
            let mut j0 = 0;
            while j0 < u {
                let j1 = (j0 + TILE).min(u);
                for di in 0..rows {
                    let i = i0 + di;
                    let orow = &mut oblk[di * u..(di + 1) * u];
                    for j in j0..j1 {
                        let c = cos_ab(a, b, v, y_norm, u, i, j);
                        let c2 = c * c;
                        let den = one_minus * c2 + lam_mu * s[i * u + j];
                        let lam = if den != 0.0 { c2 / den } else { 0.0 };
                        orow[j] = lam * c / iqr;
                    }
                }
                j0 = j1;
            }
        });
    });
    Ok(())
}

type Counts<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>,
                    Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// Null assembly for one cell-type pair (s, r) and, optionally, its mirror
/// (r, s), which shares the table M (same co-location vector, same IQR,
/// and frac_s * frac_r is symmetric in s and r).
///
/// Pair p's pseudo-pairs are (i, j) for i in its ligand's matched set and
/// j in its receptor's (`col_gene[col_ptr[p]..col_ptr[p+1]]`); `row_ptr` /
/// `row_pair` list, for every table row i, the pairs whose ligand set
/// holds i. The sweep goes row by row, so row i of M is computed once,
///
///     M[i, j] = (x[i, j] * (frac_s[i, j] * frac_r[i, j])) * scale,
///
/// and stays in cache while every pair that needs it reads it. Database
/// pairs are excluded through `dwx`, which holds NaN there: a NaN score
/// counts neither as present nor as exceeding. For direction 1,
///
///     v = ((spec_a[i] * spec_b[j]) * dw[i, j]) * M[i, j]
///
/// (the factor order of the observed score scored through the tables), and
/// for the mirror spec_a and spec_b swap. Counts per pair: exceed = #{v >=
/// obs}, present = #{v >= thr} if thr > 0 else #{v > 0}. Counting is
/// order-free, so the result does not depend on the sweep order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn assembly_sweep<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    frac_s: PyReadonlyArray2<'py, f32>,
    frac_r: PyReadonlyArray2<'py, f32>,
    scale: f64,
    dwx: PyReadonlyArray2<'py, f64>,
    spec_a: PyReadonlyArray1<'py, f64>,
    spec_b: PyReadonlyArray1<'py, f64>,
    row_ptr: PyReadonlyArray1<'py, i64>,
    row_pair: PyReadonlyArray1<'py, i32>,
    col_ptr: PyReadonlyArray1<'py, i64>,
    col_gene: PyReadonlyArray1<'py, i32>,
    obs1: PyReadonlyArray1<'py, f64>,
    obs2: PyReadonlyArray1<'py, f64>,
    thr: f64,
) -> PyResult<Counts<'py>> {
    let u = x.shape()[0];
    let x = contiguous(&x, "x", u)?;
    let fs = contiguous(&frac_s, "frac_s", u)?;
    let fr = contiguous(&frac_r, "frac_r", u)?;
    let dw = contiguous(&dwx, "dwx", u)?;
    let sa = spec_a.as_slice()?;
    let sb = spec_b.as_slice()?;
    let rp = row_ptr.as_slice()?;
    let rpair = row_pair.as_slice()?;
    let cp = col_ptr.as_slice()?;
    let cg = col_gene.as_slice()?;
    let o1 = obs1.as_slice()?;
    let o2 = obs2.as_slice()?;
    let n_pairs = o1.len();
    let mirror = !o2.is_empty();
    if sa.len() != u || sb.len() != u || rp.len() != u + 1
        || cp.len() != n_pairs + 1 || (mirror && o2.len() != n_pairs) {
        return Err(PyValueError::new_err("assembly_sweep: inconsistent sizes"));
    }
    if rpair.iter().any(|&p| p < 0 || p as usize >= n_pairs)
        || cg.iter().any(|&j| j < 0 || j as usize >= u) {
        return Err(PyValueError::new_err("assembly_sweep: index out of range"));
    }
    let zero = || (0..n_pairs).map(|_| AtomicI64::new(0)).collect::<Vec<_>>();
    let (ex1, pos1, ex2, pos2) = (zero(), zero(), zero(), zero());
    let use_thr = thr > 0.0;
    py.allow_threads(|| {
        (0..u).into_par_iter().with_min_len(8).for_each_init(
            || vec![0f64; u],
            |m_row, i| {
                let (e0, e1) = (rp[i] as usize, rp[i + 1] as usize);
                if e0 == e1 {
                    return;
                }
                let base = i * u;
                for j in 0..u {
                    let f = (fs[base + j] as f64) * (fr[base + j] as f64);
                    m_row[j] = (x[base + j] * f) * scale;
                }
                let dw_row = &dw[base..base + u];
                let (sa_i, sb_i) = (sa[i], sb[i]);
                for &p in &rpair[e0..e1] {
                    let p = p as usize;
                    let cols = &cg[cp[p] as usize..cp[p + 1] as usize];
                    let (oa, ob) = (o1[p], if mirror { o2[p] } else { 0.0 });
                    let (mut xa, mut pa, mut xb, mut pb) = (0i64, 0i64, 0i64, 0i64);
                    for &j in cols {
                        let j = j as usize;
                        let d = dw_row[j];
                        let m = m_row[j];
                        let va = ((sa_i * sb[j]) * d) * m;
                        xa += (va >= oa) as i64;
                        pa += (if use_thr { va >= thr } else { va > 0.0 }) as i64;
                        if mirror {
                            let vb = ((sb_i * sa[j]) * d) * m;
                            xb += (vb >= ob) as i64;
                            pb += (if use_thr { vb >= thr } else { vb > 0.0 }) as i64;
                        }
                    }
                    ex1[p].fetch_add(xa, Ordering::Relaxed);
                    pos1[p].fetch_add(pa, Ordering::Relaxed);
                    if mirror {
                        ex2[p].fetch_add(xb, Ordering::Relaxed);
                        pos2[p].fetch_add(pb, Ordering::Relaxed);
                    }
                }
            });
    });
    let out = |v: Vec<AtomicI64>| v.into_iter().map(|a| a.into_inner())
        .collect::<Vec<i64>>().into_pyarray_bound(py);
    Ok((out(ex1), out(pos1), out(ex2), out(pos2)))
}

// ---------------------------------------------------------------------------
// Background Gram tables, one pass over the cells (v0.15.0)
// ---------------------------------------------------------------------------
//
// The masked pseudo-pair profile of genes (i, j) in cell c is
//
//     v_ij(c) = P_i(c) P_j(c) - Q_i(c) Q_j(c) = R_i(c) P_j(c) + Q_i(c) R_j(c)
//
// with R = P on the cells where the gene is detected (raw count > 0) and 0
// elsewhere, Q = P - R. Every term on the right is >= 0. For each cell u,
//
//     M_u[i, j] = sum_c W[u, c] v_ij(c)       ((W v)(u) for every pair)
//
// and the four tables are sums over cells of entry-wise products:
//
//     G_W  = sum_u V_u * M_u      (v' W v)
//     G_W2 = sum_u M_u * M_u      (||W v||^2: no W'W graph needed)
//     G_sq = sum_u V_u * V_u,     G_dot = sum_u V_u.
//
// M_u is built from R's non-zeros only (a dense row of P or Q per detected
// gene of each neighbour), and V_u is non-zero only on the rows and columns
// of u's detected genes. All products and sums are float64 and no term
// cancels, so the tables carry ~1e-15 relative error (the four-term edge
// expansion formed its products in float32: ~1e-8). Tasks own one
// upper-triangle tile; the tables are symmetric and mirrored at the end.

const GRAM_TILE: usize = 128;

#[derive(Clone, Copy)]
struct SharedMut64(*mut f64);
unsafe impl Send for SharedMut64 {}
unsafe impl Sync for SharedMut64 {}

struct GramInputs<'a> {
    u: usize,
    nt: usize,
    p: &'a [f32],
    ri: &'a [i32],
    rd: &'a [f32],
    bptr: &'a [usize],      // per cell: first R entry of each gene tile
    wp: &'a [i64],
    wi: &'a [i32],
    wd: &'a [f64],
    order: &'a [i64],       // cells in a spatially coherent order
}

#[inline(always)]
fn gram_tile(g: &GramInputs, ti: usize, tj: usize, out: [SharedMut64; 4]) {
    let (u, nt) = (g.u, g.nt);
    let i0 = ti * GRAM_TILE;
    let ni = (i0 + GRAM_TILE).min(u) - i0;
    let j0 = tj * GRAM_TILE;
    let nj = (j0 + GRAM_TILE).min(u) - j0;
    // m[a][b] accumulates sum_c w R_{i c} P_{j c} and then becomes M_u;
    // t2[b][a] = sum_c w R_{j c} Q_{i c} (row updates; transposed into m).
    // Both are zero at the top of every cell: cleared as they are consumed.
    let mut m = vec![0f64; ni * nj];
    let mut t2 = vec![0f64; nj * ni];
    let (mut gw, mut gw2) = (vec![0f64; ni * nj], vec![0f64; ni * nj]);
    let (mut gsq, mut gdot) = (vec![0f64; ni * nj], vec![0f64; ni * nj]);
    let mut pj = vec![0f64; nj];
    let mut qi = vec![0f64; ni];
    let mut det_i = vec![false; ni];
    for &cu in g.order {
        let cu = cu as usize;
        let (mut any_m, mut any_t2) = (false, false);
        for e in g.wp[cu] as usize..g.wp[cu + 1] as usize {
            let c = g.wi[e] as usize;
            let w = g.wd[e];
            let bp = &g.bptr[c * (nt + 1)..(c + 1) * (nt + 1)];
            let (fi0, fi1, fj0, fj1) = (bp[ti], bp[ti + 1], bp[tj], bp[tj + 1]);
            if fi0 < fi1 {
                any_m = true;
                for (x, &pv) in pj.iter_mut().zip(&g.p[c * u + j0..c * u + j0 + nj]) {
                    *x = pv as f64;
                }
                for f in fi0..fi1 {
                    let a = g.ri[f] as usize - i0;
                    let s = w * g.rd[f] as f64;
                    for (x, &pv) in m[a * nj..(a + 1) * nj].iter_mut().zip(pj.iter()) {
                        *x += s * pv;
                    }
                }
            }
            if fj0 < fj1 {
                any_t2 = true;
                // Q_c on the tile's rows: P_c with c's detected genes zeroed
                for (q, &pv) in qi.iter_mut().zip(&g.p[c * u + i0..c * u + i0 + ni]) {
                    *q = pv as f64;
                }
                for f in fi0..fi1 {
                    qi[g.ri[f] as usize - i0] = 0.0;
                }
                for f in fj0..fj1 {
                    let b = g.ri[f] as usize - j0;
                    let s = w * g.rd[f] as f64;
                    for (x, &q) in t2[b * ni..(b + 1) * ni].iter_mut().zip(qi.iter()) {
                        *x += s * q;
                    }
                }
            }
        }
        if any_t2 {
            // m += t2', in 16 x 16 blocks so both sides stay in L1
            const B: usize = 16;
            for a0 in (0..ni).step_by(B) {
                for b0 in (0..nj).step_by(B) {
                    for b in b0..(b0 + B).min(nj) {
                        for a in a0..(a0 + B).min(ni) {
                            m[a * nj + b] += t2[b * ni + a];
                            t2[b * ni + a] = 0.0;
                        }
                    }
                }
            }
        }
        // V_u: rows of u's detected genes, V = R_iu P_ju; on the other rows,
        // columns of u's detected genes, V = P_iu R_ju (Q_iu = P_iu there)
        let bp = &g.bptr[cu * (g.nt + 1)..(cu + 1) * (g.nt + 1)];
        let (fi0, fi1, fj0, fj1) = (bp[ti], bp[ti + 1], bp[tj], bp[tj + 1]);
        if fi0 < fi1 || fj0 < fj1 {
            let pu_i = &g.p[cu * u + i0..cu * u + i0 + ni];
            let pu_j = &g.p[cu * u + j0..cu * u + j0 + nj];
            det_i.fill(false);
            for f in fi0..fi1 {
                let a = g.ri[f] as usize - i0;
                det_i[a] = true;
                let r = g.rd[f] as f64;
                let row = a * nj..(a + 1) * nj;
                for (((gwk, gsqk), gdotk), (&mk, &pv)) in gw[row.clone()].iter_mut()
                    .zip(gsq[row.clone()].iter_mut()).zip(gdot[row.clone()].iter_mut())
                    .zip(m[row].iter().zip(pu_j)) {
                    let v = r * pv as f64;
                    *gwk += v * mk;
                    *gsqk += v * v;
                    *gdotk += v;
                }
            }
            for f in fj0..fj1 {
                let b = g.ri[f] as usize - j0;
                let r = g.rd[f] as f64;
                for a in 0..ni {
                    if det_i[a] {
                        continue;
                    }
                    let k = a * nj + b;
                    let v = pu_i[a] as f64 * r;
                    gw[k] += v * m[k];
                    gsq[k] += v * v;
                    gdot[k] += v;
                }
            }
        }
        if any_m || any_t2 {
            // G_W2 += M * M, clearing m for the next cell
            for (x, acc) in m.iter_mut().zip(gw2.iter_mut()) {
                *acc += *x * *x;
                *x = 0.0;
            }
        }
    }
    for (dst, src) in out.iter().zip([&gw, &gw2, &gsq, &gdot]) {
        for a in 0..ni {
            // SAFETY: tile (ti, tj) of the output belongs to this task alone
            let row = unsafe {
                std::slice::from_raw_parts_mut(dst.0.add((i0 + a) * u + j0), nj)
            };
            row.copy_from_slice(&src[a * nj..(a + 1) * nj]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gram_tile_avx2(g: &GramInputs, ti: usize, tj: usize, out: [SharedMut64; 4]) {
    gram_tile(g, ti, tj, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gram_tile_avx512(g: &GramInputs, ti: usize, tj: usize, out: [SharedMut64; 4]) {
    gram_tile(g, ti, tj, out)
}

type Grams<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>,
                   Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// The four background Gram tables (G_W, G_W2, G_sq, G_dot), U x U
/// float64, from P (cells x U, float32), R = P on detected entries (CSR,
/// cells x U, genes ascending per cell) and the row-normalised spatial
/// graph W (CSR, cells x cells). See the notes above. `cell_order` is a
/// permutation of the cells: visiting them in a spatially coherent order
/// (reverse Cuthill-McKee of W) keeps the neighbours' rows of P in cache.
/// It changes only the summation order (float64 rounding).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn background_grams<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f32>,
    r_indptr: PyReadonlyArray1<'py, i64>,
    r_indices: PyReadonlyArray1<'py, i32>,
    r_data: PyReadonlyArray1<'py, f32>,
    w_indptr: PyReadonlyArray1<'py, i64>,
    w_indices: PyReadonlyArray1<'py, i32>,
    w_data: PyReadonlyArray1<'py, f64>,
    cell_order: PyReadonlyArray1<'py, i64>,
) -> PyResult<Grams<'py>> {
    let (n, u) = (p.shape()[0], p.shape()[1]);
    if !p.is_c_contiguous() {
        return Err(PyValueError::new_err("p must be C-contiguous"));
    }
    let pf = p.as_slice()?;
    let (rp, ri, rd) = (r_indptr.as_slice()?, r_indices.as_slice()?, r_data.as_slice()?);
    let (wp, wi, wd) = (w_indptr.as_slice()?, w_indices.as_slice()?, w_data.as_slice()?);
    let order = cell_order.as_slice()?;
    if rp.len() != n + 1 || ri.len() != rd.len() || rp[n] as usize != ri.len()
        || wp.len() != n + 1 || wi.len() != wd.len() || wp[n] as usize != wi.len()
        || order.len() != n {
        return Err(PyValueError::new_err("background_grams: inconsistent sizes"));
    }
    if wi.iter().any(|&c| c < 0 || c as usize >= n) {
        return Err(PyValueError::new_err("background_grams: graph index out of range"));
    }
    let mut seen = vec![false; n];
    for &c in order {
        if c < 0 || c as usize >= n || seen[c as usize] {
            return Err(PyValueError::new_err(
                "background_grams: cell_order must be a permutation of the cells"));
        }
        seen[c as usize] = true;
    }
    let nt = u.div_ceil(GRAM_TILE);
    let mut bptr = vec![0usize; n * (nt + 1)];
    let ok = py.allow_threads(|| {
        bptr.par_chunks_mut(nt + 1).enumerate().map(|(c, bp)| {
            let (s, e) = (rp[c] as usize, rp[c + 1] as usize);
            let row = &ri[s..e];
            if row.windows(2).any(|w| w[0] >= w[1])
                || row.iter().any(|&j| j < 0 || j as usize >= u) {
                return false;
            }
            let mut f = s;
            for (t, slot) in bp.iter_mut().enumerate() {
                while f < e && (ri[f] as usize) < t * GRAM_TILE {
                    f += 1;
                }
                *slot = f;
            }
            true
        }).all(|x| x)
    });
    if !ok {
        return Err(PyValueError::new_err(
            "background_grams: R's gene indices must be sorted and in range"));
    }
    let g = GramInputs { u, nt, p: pf, ri, rd, bptr: &bptr, wp, wi, wd, order };
    let mut outs: Vec<Vec<f64>> = (0..4).map(|_| vec![0f64; u * u]).collect();
    let ptrs: Vec<SharedMut64> = outs.iter_mut().map(|o| SharedMut64(o.as_mut_ptr())).collect();
    let ptrs = [ptrs[0], ptrs[1], ptrs[2], ptrs[3]];
    let tiles: Vec<(usize, usize)> = (0..nt)
        .flat_map(|ti| (ti..nt).map(move |tj| (ti, tj))).collect();
    #[cfg(target_arch = "x86_64")]
    let level = if is_x86_feature_detected!("avx512f") { 2 }
                else if is_x86_feature_detected!("avx2") { 1 } else { 0 };
    #[cfg(not(target_arch = "x86_64"))]
    let level = 0;
    py.allow_threads(|| {
        // big tiles first would not help: every tile sweeps all cells
        tiles.par_iter().with_max_len(1).for_each(|&(ti, tj)| match level {
            // SAFETY: the CPU supports the feature (checked above)
            #[cfg(target_arch = "x86_64")]
            2 => unsafe { gram_tile_avx512(&g, ti, tj, ptrs) },
            #[cfg(target_arch = "x86_64")]
            1 => unsafe { gram_tile_avx2(&g, ti, tj, ptrs) },
            _ => gram_tile(&g, ti, tj, ptrs),
        });
        // mirror the upper triangle (tables are symmetric)
        for o in outs.iter_mut() {
            let src = SharedMut64(o.as_mut_ptr());
            (0..u).into_par_iter().for_each(|i| {
                let s = src;
                for j in 0..i {
                    // SAFETY: row i's lower part is written only here, and
                    // reads touch the upper triangle, which no task writes
                    unsafe { *s.0.add(i * u + j) = *s.0.add(j * u + i); }
                }
            });
        }
    });
    let mut arrs = outs.into_iter().map(|o| {
        Array2::from_shape_vec((u, u), o)
            .map(|a| a.into_pyarray_bound(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    });
    let mut next = || arrs.next().unwrap();
    Ok((next()?, next()?, next()?, next()?))
}

/// Size rayon's global pool before any kernel runs, so the Rust threads
/// follow the same count as LARIS's Python thread pool (the CPUs this
/// process may use, or RAYON_NUM_THREADS / OMP_NUM_THREADS). Returns false
/// when the pool already exists: rayon builds it once per process.
#[pyfunction]
fn init_threads(n: usize) -> bool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .build_global()
        .is_ok()
}

/// Threads in rayon's global pool (diagnostics and tests).
#[pyfunction]
fn current_num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _laris(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_threads, m)?)?;
    m.add_function(wrap_pyfunction!(current_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(ctc_numerator_ab, m)?)?;
    m.add_function(wrap_pyfunction!(ctc_cos_numerator, m)?)?;
    m.add_function(wrap_pyfunction!(cos_sq_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(coloc_x, m)?)?;
    m.add_function(wrap_pyfunction!(assembly_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(background_grams, m)?)?;
    Ok(())
}
