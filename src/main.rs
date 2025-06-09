use cp_proj::tensor::Tensor;
use std::time::Instant;

fn count_diff(a: &Tensor, b: &Tensor, epsilon: f32) {
    assert_eq!(a.data.len(), b.data.len(), "Tensor data lengths must match");

    let mut max = 0 as f32;
    let mut count = 0;

    for i in 0..a.data.len() {
        if (a.data[i] - b.data[i]).abs() > epsilon {
            count += 1;
            max = max.max((a.data[i] - b.data[i]).abs());
        }
    }

    println!("max: {}, count: {}", max, count);

}

fn main() {

    let mat1 = Tensor::random(1024, 28*28, 42);
    let mat2 = Tensor::random(28*28, 2, 24);

    let start = Instant::now();
    let r1 = mat1.mul_simd_parallel(&mat2, 6);
    let simd_duration = start.elapsed();
    println!("parallel SIMD duration: {:?}", simd_duration);

    let start = Instant::now();
    let r2 = mat1.mul_seq(&mat2);
    let mul_seq_duration = start.elapsed();
    println!("mul_vec_parallel duration: {:?}", mul_seq_duration);

    println!("Equal: {}", r1 == r2);

    println!("mul_ vs mul_seq: {:.2}x", mul_seq_duration.as_secs_f64() / simd_duration.as_secs_f64());

    println!("Dims r1: {}x{}", r1.rows, r1.cols);
    println!("Dims r1: {}x{}", r2.rows, r2.cols);
    count_diff(&r1, &r2, 0.0001);

}