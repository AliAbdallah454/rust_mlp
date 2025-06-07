use cp_proj::Tensor;

fn main() {

    let size = 1_000;

    let mat1 = Tensor::random(size, size, 42);
    let mat2 = Tensor::random(size, size, 24);

    // Run sequential version first
    let start_seq = std::time::Instant::now();
    let _res_seq = mat1.mul_seq(&mat2);
    let duration_seq = start_seq.elapsed();
    println!("Sequential multiplication took: {:.3?}", duration_seq);

    // Test different thread counts
    let thread_counts = vec![4, 8, 12];
    let mut durations = Vec::new();

    for &threads in &thread_counts {
        let start_par = std::time::Instant::now();
        let _res_par = mat1.mul_par(&mat2, threads);
        let duration_par = start_par.elapsed();
        durations.push(duration_par);
        println!("Parallel multiplication ({} threads) took: {:.3?}", threads, duration_par);
    }

    // Calculate and print speedups
    println!("\nSpeedups relative to sequential:");
    for (i, duration) in durations.iter().enumerate() {
        let speedup = duration_seq.as_secs_f64() / duration.as_secs_f64();
        println!("{} threads: {:.2}x", thread_counts[i], speedup);
    }
}