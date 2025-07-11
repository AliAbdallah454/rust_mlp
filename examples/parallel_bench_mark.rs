use std::fs::File;
use std::io::Write;
use std::time::Instant;
use cp_proj::tensor::{Tensor, ExecutionMode};

fn main() {
    let size = 32;
    let seed = 123;

    let a = Tensor::random(size, size, seed);
    let b = Tensor::random(size, size, seed + 1);

    let mut file = File::create("parallel_benchmark_32.csv").expect("Unable to create file");
    writeln!(file, "Mode,Threads,Time(s),Speedup_vs_Sequential").unwrap();

    // Baseline: Sequential
    let start = Instant::now();
    let baseline = a.mul(&b, ExecutionMode::Sequential);
    let time_seq = start.elapsed().as_secs_f64();
    writeln!(file, "Sequential,1,{:.6},1.0", time_seq).unwrap();

    for threads in 1..=6 {
        let start = Instant::now();
        let parallel = a.mul(&b, ExecutionMode::Parallel(threads));
        let time_par = start.elapsed().as_secs_f64();

        assert!(baseline == parallel, "Mismatch in results between Sequential and Parallel({})", threads);

        writeln!(
            file,
            "Parallel,{},{:.6},{:.4}",
            threads,
            time_par,
            time_seq / time_par
        ).unwrap();
    }

    println!("Benchmark results written to parallel_benchmark_32.csv");
}
