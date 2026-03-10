//! Optional BLAS backend for f32 GEMM.
//!
//! When the `blas` feature is enabled, delegates large f32 matrix multiplications
//! to the system BLAS (e.g., OpenBLAS, MKL) via `cblas_sgemm`. This gives
//! near-optimal performance on native targets while keeping the pure-Rust
//! kernels available for WASI and other targets.

use std::mem::MaybeUninit;

use rten_tensor::MatrixLayout;

use crate::{BiasVector, GemmInputA, GemmInputB};

// FFI binding to CBLAS
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,    // CblasRowMajor = 101
        transa: i32,   // CblasNoTrans = 111
        transb: i32,   // CblasNoTrans = 111
        m: i32,        // rows of A / output
        n: i32,        // cols of B / output
        k: i32,        // cols of A / rows of B
        alpha: f32,
        a: *const f32,
        lda: i32,      // leading dimension of A
        b: *const f32,
        ldb: i32,      // leading dimension of B
        beta: f32,
        c: *mut f32,
        ldc: i32,      // leading dimension of C
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// Try to execute a GEMM using the system BLAS.
///
/// Returns `Some(Ok(()))` if handled by BLAS, `None` if it should fall through.
/// Only handles f32 unpacked contiguous inputs with no quantization.
pub(crate) fn try_blas_sgemm(
    out_data: &mut [MaybeUninit<f32>],
    a: GemmInputA<f32>,
    b: GemmInputB<f32>,
    alpha: f32,
    beta: f32,
    bias: Option<BiasVector<f32>>,
) -> Option<super::GemmResult<()>> {
    // Only handle unpacked f32 matrices
    let (a_mat, b_mat) = match (a, b) {
        (GemmInputA::Unpacked(a), GemmInputB::Unpacked(b)) => (a, b),
        _ => return None,
    };

    let m = a_mat.rows();
    let k = a_mat.cols();
    let n = b_mat.cols();

    // BLAS needs contiguous data. Check via row_stride.
    // Row-major: row_stride == cols, col_stride == 1
    let a_data = a_mat.data()?;
    let a_row_stride = a_mat.row_stride();
    let (a_ptr, lda, transa) = if a_mat.col_stride() == 1 {
        (a_data.as_ptr(), a_row_stride as i32, CBLAS_NO_TRANS)
    } else if a_row_stride == 1 {
        // Column-major (transposed)
        (a_data.as_ptr(), a_mat.col_stride() as i32, CBLAS_TRANS)
    } else {
        return None;
    };

    let b_data = b_mat.data()?;
    let b_row_stride = b_mat.row_stride();
    let (b_ptr, ldb, transb) = if b_mat.col_stride() == 1 {
        (b_data.as_ptr(), b_row_stride as i32, CBLAS_NO_TRANS)
    } else if b_row_stride == 1 {
        (b_data.as_ptr(), b_mat.col_stride() as i32, CBLAS_TRANS)
    } else {
        return None;
    };

    let c_ptr = out_data.as_mut_ptr() as *mut f32;
    let ldc = n as i32;

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            transa,
            transb,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            beta,
            c_ptr,
            ldc,
        );
    }

    // Apply bias if present
    if let Some(bias) = bias {
        match bias {
            BiasVector::Row(row_bias) => {
                for i in 0..m {
                    for j in 0..n {
                        unsafe {
                            let ptr = c_ptr.add(i * n + j);
                            *ptr += row_bias[j];
                        }
                    }
                }
            }
            BiasVector::Column(col_bias) => {
                for i in 0..m {
                    for j in 0..n {
                        unsafe {
                            let ptr = c_ptr.add(i * n + j);
                            *ptr += col_bias[i];
                        }
                    }
                }
            }
        }
    }

    Some(Ok(()))
}
