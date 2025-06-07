use cp_proj::tensor::Tensor;
use std::time::Instant;

fn main() {
    let sizes = vec![16, 32, 64, 128, 256, 512];
    let num_tries = 100;
    let image_size = 28 * 28;
    
    println!("Comparing sequential vs parallel Hadamard implementations (averaged over {} tries)\n", num_tries);
    println!("Batch Size\tSequential (ms)\tParallel (ms)\tSpeedup");
    println!("--------------------------------------------------------");

    for &batch_size in &sizes {
        // Create random tensors of size batch_size x (28*28)
        let tensor1 = Tensor::random(batch_size, image_size, 42);
        let tensor2 = Tensor::random(batch_size, image_size, 24);
        
        let mut total_seq_duration = 0.0;
        let mut total_par_duration = 0.0;
        
        for _ in 0..num_tries {
            // Benchmark sequential implementation
            let seq_start = Instant::now();
            let _seq_result = tensor1.hadamard(&tensor2);
            total_seq_duration += seq_start.elapsed().as_secs_f64();
            
            // Benchmark parallel implementation
            let par_start = Instant::now();
            let _par_result = tensor1.hadamard_par(&tensor2);
            total_par_duration += par_start.elapsed().as_secs_f64();
        }
        
        // Calculate averages
        let avg_seq_duration = total_seq_duration / num_tries as f64;
        let avg_par_duration = total_par_duration / num_tries as f64;
        
        // Calculate speedup
        let speedup = avg_seq_duration / avg_par_duration;
        
        println!("{:<10}\t{:.2}\t\t{:.2}\t\t{:.2}x", 
            batch_size,
            avg_seq_duration * 1000.0,
            avg_par_duration * 1000.0,
            speedup);
    }
}
