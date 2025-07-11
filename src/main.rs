use std::time::Instant;
use std::fs::File;
use std::io::Write;

use cp_proj::tensor::ExecutionMode;
use cp_proj::tensor::Tensor;

fn main() {
    let sizes = vec![32, 64, 128, 256, 512, 1024, 2048, 4096];
    let seed = 42u64;

    println!("{:<8} | {:<12} | {:<12} | {:<12} | {:<12}", "Size", "Sequential", "Parallel", "SIMD", "ParSIMD");
    println!("{}", "-".repeat(66));

    let mut file = File::create("tensor_benchmark.csv").expect("Failed to create CSV file");
    writeln!(file, "Size,Sequential,Parallel Speedup,SIMD Speedup,ParallelSIMD Speedup").unwrap();

    for &size in &sizes {
        let a = Tensor::random(size, size, seed);
        let b = Tensor::random(size, size, seed + 1);

        let start = Instant::now();
        let baseline = a.mul(&b, ExecutionMode::Sequential);
        let duration_seq = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let parallel = a.mul(&b, ExecutionMode::Parallel(6));
        let duration_par = start.elapsed().as_secs_f64();
        assert!(baseline == parallel);

        let start = Instant::now();
        let simd = a.mul(&b, ExecutionMode::SIMD);
        let duration_simd = start.elapsed().as_secs_f64();
        assert!(baseline == simd);

        let start = Instant::now();
        let parsimd = a.mul(&b, ExecutionMode::ParallelSIMD(6));
        let duration_parsimd = start.elapsed().as_secs_f64();
        assert!(baseline == parsimd);

        let speedup_par = duration_seq / duration_par;
        let speedup_simd = duration_seq / duration_simd;
        let speedup_parsimd = duration_seq / duration_parsimd;

        println!("{:<8} | {:<12.5} | {:<12.5} | {:<12.5} | {:<12.5}",
            size, duration_seq, speedup_par, speedup_simd, speedup_parsimd);

        writeln!(file, "{},{:.5},{:.5},{:.5},{:.5}",
            size, duration_seq, speedup_par, speedup_simd, speedup_parsimd).unwrap();
    }
}