//! RTen prepacks MatMulInteger weights without their zero points. The
//! vendored fix (vendor/rten/src/ops/matmul.rs `matmul_integer`) must apply
//! nonzero weight zero points on the prepacked path, which is what makes the
//! int8 Nemotron encoder both fast and correct.
use std::{path::Path, sync::Arc};

use rten::{ModelOptions, RunOptions, ThreadPool, Value};

const DEPTH: usize = 64;
const COLUMNS: usize = 7;

fn b(k: usize, c: usize) -> i64 {
    ((31 * k + 17 * c) % 256) as i64 - 128
}

#[test]
fn prepacked_matmul_integer_applies_weight_zero_points() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let pool = Arc::new(ThreadPool::with_num_threads(2).unwrap());
    let options = RunOptions::default().with_thread_pool(Some(pool.clone()));
    for (name, zero) in [
        ("scalar", [-7_i64; COLUMNS]),
        ("column", [3, -2, 0, 9, -128, 127, 1]),
    ] {
        let mut loader = ModelOptions::with_all_ops();
        loader.enable_optimization(false);
        let mut model = loader
            .load(
                std::fs::read(fixtures.join(format!("matmul_integer_b_zero_{name}.onnx"))).unwrap(),
            )
            .unwrap();
        model.prepack_weights(&pool);
        for rows in [1_usize, 2, 5, 41] {
            let a: Vec<u8> = (0..rows * DEPTH)
                .map(|i| ((i * 37 + 11) % 256) as u8)
                .collect();
            let a_zero = 131_u8;
            let [y] = model
                .run_n(
                    vec![
                        (
                            model.node_id("a").unwrap(),
                            Value::from_shape(&[rows, DEPTH], a.clone()).unwrap().into(),
                        ),
                        (
                            model.node_id("a_zero").unwrap(),
                            Value::from_shape(&[] as &[usize], vec![a_zero])
                                .unwrap()
                                .into(),
                        ),
                    ],
                    [model.node_id("y").unwrap()],
                    Some(options.clone()),
                )
                .unwrap();
            let Value::Int32Tensor(y) = y else {
                panic!("MatMulInteger must produce i32");
            };
            for row in 0..rows {
                for column in 0..COLUMNS {
                    let expected: i64 = (0..DEPTH)
                        .map(|k| {
                            (i64::from(a[row * DEPTH + k]) - i64::from(a_zero))
                                * (b(k, column) - zero[column])
                        })
                        .sum();
                    assert_eq!(
                        i64::from(y[[row, column]]),
                        expected,
                        "{name} zero point, {rows} rows, row {row} column {column}"
                    );
                }
            }
        }
    }
}
