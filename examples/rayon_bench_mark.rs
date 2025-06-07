use rayon::prelude::*;
use std::time::Instant;
use rand::Rng;

fn main() {
    let size = 10_000_000;
    let num_tries = 1000;
    
    println!("Comparing sequential vs parallel squaring (averaged over {} tries)\n", num_tries);
    println!("Size\t\tSequential (ms)\tParallel (ms)\tSpeedup");
    println!("--------------------------------------------------------");

    // Create a random vector
    let mut rng = rand::thread_rng();
    let vec: Vec<f64> = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    let mut total_seq_duration = 0.0;
    let mut total_par_duration = 0.0;
    
    for _ in 0..num_tries {
        // Benchmark sequential implementation
        let seq_start = Instant::now();
        let _seq_result: Vec<f64> = vec.iter().map(|&x| x * x).collect();
        total_seq_duration += seq_start.elapsed().as_secs_f64();
        
        // Benchmark parallel implementation using rayon
        let par_start = Instant::now();
        let _par_result: Vec<f64> = vec.par_iter().map(|&x| x * x).collect();
        total_par_duration += par_start.elapsed().as_secs_f64();
    }
    
    // Calculate averages
    let avg_seq_duration = total_seq_duration / num_tries as f64;
    let avg_par_duration = total_par_duration / num_tries as f64;
    
    // Calculate speedup
    let speedup = avg_seq_duration / avg_par_duration;
    
    println!("{:<10}\t{:.2}\t\t{:.2}\t\t{:.2}x", 
        size,
        avg_seq_duration * 1000.0,
        avg_par_duration * 1000.0,
        speedup);
}
