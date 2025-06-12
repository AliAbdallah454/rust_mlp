use cp_proj::tensor::{ExecutionMode, Tensor};

fn main() {

    let size = 500;

    let mat1 = Tensor::random(size, size, 42);
    let mat2 = Tensor::random(size, size, 24);

    let start_seq = std::time::Instant::now();
    let _res_seq = mat1.mul_seq(&mat2);
    let duration_seq = start_seq.elapsed();
    println!("Sequential multiplication took: {:.3?}", duration_seq);

    let modes = vec![
        ExecutionMode::Parallel,
        ExecutionMode::SIMD,
        ExecutionMode::ParallelSIMD
    ];
    let mut durations = Vec::new();

    for &mode in &modes {
        let start_par = std::time::Instant::now();
        let _res_par = mat1.mul(&mat2, mode);
        let duration_par = start_par.elapsed();
        durations.push(duration_par);
        println!("{:?} took: {:.3?}", mode, duration_par);
    }

    println!("\nSpeedups relative to sequential:");
    for (i, duration) in durations.iter().enumerate() {
        let speedup = duration_seq.as_secs_f64() / duration.as_secs_f64();
        println!("{:?} threads: {:.2}x", modes[i], speedup);
    }
}