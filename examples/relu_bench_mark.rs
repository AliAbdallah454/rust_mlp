use cp_proj::tensor::Tensor;
use std::time::Instant;

fn main() {
    let sizes = vec![10_000, 50_000, 100_000, 500_000];
    let num_tries = 100;
    
    println!("Comparing sequential vs parallel ReLU implementations (averaged over {} tries)\n", num_tries);
    println!("Size\t\tSequential (ms)\tParallel (ms)\tSpeedup");
    println!("--------------------------------------------------------");

    for &size in &sizes {
        // Create a random tensor of size x 1
        let tensor = Tensor::random(size, 1, 42);
        
        let mut total_seq_duration = 0.0;
        let mut total_par_duration = 0.0;
        
        for _ in 0..num_tries {
            // Benchmark sequential implementation
            let seq_start = Instant::now();
            let _seq_result = tensor.relu();
            total_seq_duration += seq_start.elapsed().as_secs_f64();
            
            // Benchmark parallel implementation
            let par_start = Instant::now();
            let _par_result = tensor.relu_rayon();
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
}
