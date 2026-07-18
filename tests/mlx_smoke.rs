#![cfg(feature = "mlx")]

use mlx_rs::Array;

#[test]
fn metal_gpu_matmul_works() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);
    let c = mlx_rs::ops::matmul(&a, &b).expect("matmul");
    c.eval().expect("eval");
    let vals: Vec<f32> = c.as_slice().to_vec();
    assert_eq!(vals, vec![19.0, 22.0, 43.0, 50.0]);
}
