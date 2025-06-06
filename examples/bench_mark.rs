use std::time::Instant;
use cp_proj::Tensor;

// Assuming your Tensor module is available
// mod tensor; // Uncomment if your Tensor is in a separate module
// use tensor::Tensor;

fn benchmark_function<F>(name: &str, func: F, iterations: u32) -> f64 
where 
    F: Fn() -> ()
{
    let start = Instant::now();
    for _ in 0..iterations {
        func();
    }
    let duration = start.elapsed();
    let avg_time = duration.as_nanos() as f64 / iterations as f64 / 1_000_000.0; // Convert to milliseconds
    println!("{}: {:.4} ms (avg over {} iterations)", name, avg_time, iterations);
    avg_time
}

fn main() {
    println!("=== Tensor Performance Benchmark ===\n");
    
    let iterations = 100;
    let large_size = 500;
    let medium_size = 200;
    let small_size = 100;
    
    // Test 1: Addition
    println!("1. TENSOR ADDITION");
    let a_seq = Tensor::new_with_concurrent(
        (0..large_size * large_size).map(|i| i as f64).collect(),
        large_size, large_size, false
    );
    let b_seq = Tensor::new_with_concurrent(
        (0..large_size * large_size).map(|i| (i * 2) as f64).collect(),
        large_size, large_size, false
    );
    let a_par = Tensor::new_with_concurrent(
        (0..large_size * large_size).map(|i| i as f64).collect(),
        large_size, large_size, true
    );
    let b_par = Tensor::new_with_concurrent(
        (0..large_size * large_size).map(|i| (i * 2) as f64).collect(),
        large_size, large_size, true
    );
    
    let seq_time = benchmark_function("  Sequential Add", || {
        let _ = &a_seq + &b_seq;
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Add", || {
        let _ = &a_par + &b_par;
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 2: Subtraction
    println!("2. TENSOR SUBTRACTION");
    let seq_time = benchmark_function("  Sequential Sub", || {
        let _ = &a_seq - &b_seq;
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Sub", || {
        let _ = &a_par - &b_par;
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 3: Matrix Multiplication
    println!("3. MATRIX MULTIPLICATION");
    let m1_seq = Tensor::new_with_concurrent(
        (0..medium_size * medium_size).map(|i| (i as f64) * 0.01).collect(),
        medium_size, medium_size, false
    );
    let m2_seq = Tensor::new_with_concurrent(
        (0..medium_size * medium_size).map(|i| (i as f64) * 0.02).collect(),
        medium_size, medium_size, false
    );
    let m1_par = Tensor::new_with_concurrent(
        (0..medium_size * medium_size).map(|i| (i as f64) * 0.01).collect(),
        medium_size, medium_size, true
    );
    let m2_par = Tensor::new_with_concurrent(
        (0..medium_size * medium_size).map(|i| (i as f64) * 0.02).collect(),
        medium_size, medium_size, true
    );
    
    let seq_time = benchmark_function("  Sequential Mul", || {
        let _ = m1_seq.mul(&m2_seq);
    }, 10); // Fewer iterations for expensive operation
    
    let par_time = benchmark_function("  Parallel Mul", || {
        let _ = m1_par.mul(&m2_par);
    }, 10);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 4: Hadamard (Element-wise) Multiplication
    println!("4. HADAMARD MULTIPLICATION");
    let seq_time = benchmark_function("  Sequential Hadamard", || {
        let _ = a_seq.hadamard(&b_seq);
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Hadamard", || {
        let _ = a_par.hadamard(&b_par);
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 5: Square
    println!("5. TENSOR SQUARE");
    let seq_time = benchmark_function("  Sequential Square", || {
        let _ = a_seq.square();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Square", || {
        let _ = a_par.square();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 6: ReLU
    println!("6. RELU ACTIVATION");
    let seq_time = benchmark_function("  Sequential ReLU", || {
        let _ = a_seq.relu();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel ReLU", || {
        let _ = a_par.relu();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 7: ReLU Derivative
    println!("7. RELU DERIVATIVE");
    let seq_time = benchmark_function("  Sequential ReLU Derivative", || {
        let _ = a_seq.relu_derivative();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel ReLU Derivative", || {
        let _ = a_par.relu_derivative();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 8: Sigmoid
    println!("8. SIGMOID ACTIVATION");
    let seq_time = benchmark_function("  Sequential Sigmoid", || {
        let _ = a_seq.sigmoid();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Sigmoid", || {
        let _ = a_par.sigmoid();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 9: Sigmoid Derivative
    println!("9. SIGMOID DERIVATIVE");
    let seq_time = benchmark_function("  Sequential Sigmoid Derivative", || {
        let _ = a_seq.sigmoid_derivative();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Sigmoid Derivative", || {
        let _ = a_par.sigmoid_derivative();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 10: Tanh
    println!("10. TANH ACTIVATION");
    let seq_time = benchmark_function("  Sequential Tanh", || {
        let _ = a_seq.tanh();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Tanh", || {
        let _ = a_par.tanh();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 11: Tanh Derivative
    println!("11. TANH DERIVATIVE");
    let seq_time = benchmark_function("  Sequential Tanh Derivative", || {
        let _ = a_seq.tanh_derivative();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Tanh Derivative", || {
        let _ = a_par.tanh_derivative();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 12: Softmax (Column vector)
    println!("12. SOFTMAX ACTIVATION");
    let col_seq = Tensor::new_with_concurrent(
        (0..large_size).map(|i| (i as f64) * 0.01).collect(),
        large_size, 1, false
    );
    let col_par = Tensor::new_with_concurrent(
        (0..large_size).map(|i| (i as f64) * 0.01).collect(),
        large_size, 1, true
    );
    
    let seq_time = benchmark_function("  Sequential Softmax", || {
        let _ = col_seq.softmax();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Softmax", || {
        let _ = col_par.softmax();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 13: Softmax Derivative
    println!("13. SOFTMAX DERIVATIVE");
    let seq_time = benchmark_function("  Sequential Softmax Derivative", || {
        let _ = col_seq.softmax_derivative();
    }, 10); // Fewer iterations for expensive operation
    
    let par_time = benchmark_function("  Parallel Softmax Derivative", || {
        let _ = col_par.softmax_derivative();
    }, 10);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 14: Transpose
    println!("14. TRANSPOSE");
    let seq_time = benchmark_function("  Sequential Transpose", || {
        let _ = a_seq.transpose();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Transpose", || {
        let _ = a_par.transpose();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 15: Scale
    println!("15. SCALE");
    let seq_time = benchmark_function("  Sequential Scale", || {
        let _ = a_seq.scale(2.5);
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Scale", || {
        let _ = a_par.scale(2.5);
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 16: Sum
    println!("16. SUM");
    let seq_time = benchmark_function("  Sequential Sum", || {
        let _ = a_seq.sum();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Sum", || {
        let _ = a_par.sum();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 17: MSE Loss
    println!("17. MSE LOSS");
    let seq_time = benchmark_function("  Sequential MSE Loss", || {
        let _ = a_seq.mse_loss(&b_seq);
    }, iterations);
    
    let par_time = benchmark_function("  Parallel MSE Loss", || {
        let _ = a_par.mse_loss(&b_par);
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 18: MSE Loss Derivative
    println!("18. MSE LOSS DERIVATIVE");
    let seq_time = benchmark_function("  Sequential MSE Loss Derivative", || {
        let _ = a_seq.mse_loss_derivative(&b_seq);
    }, iterations);
    
    let par_time = benchmark_function("  Parallel MSE Loss Derivative", || {
        let _ = a_par.mse_loss_derivative(&b_par);
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 19: Categorical Cross Entropy
    println!("19. CATEGORICAL CROSS ENTROPY");
    let pred_seq = Tensor::new_with_concurrent(
        (0..small_size).map(|i| (i as f64 + 1.0) / (small_size as f64 + 1.0)).collect(),
        small_size, 1, false
    );
    let target_seq = Tensor::new_with_concurrent(
        (0..small_size).map(|i| if i == 10 { 1.0 } else { 0.0 }).collect(),
        small_size, 1, false
    );
    let pred_par = Tensor::new_with_concurrent(
        (0..small_size).map(|i| (i as f64 + 1.0) / (small_size as f64 + 1.0)).collect(),
        small_size, 1, true
    );
    let target_par = Tensor::new_with_concurrent(
        (0..small_size).map(|i| if i == 10 { 1.0 } else { 0.0 }).collect(),
        small_size, 1, true
    );
    
    let seq_time = benchmark_function("  Sequential Categorical Cross Entropy", || {
        let _ = pred_seq.categorical_cross_entropy(&target_seq);
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Categorical Cross Entropy", || {
        let _ = pred_par.categorical_cross_entropy(&target_par);
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 20: Categorical Cross Entropy Derivative
    println!("20. CATEGORICAL CROSS ENTROPY DERIVATIVE");
    let seq_time = benchmark_function("  Sequential CCE Derivative", || {
        let _ = pred_seq.categorical_cross_entropy_derivative(&target_seq);
    }, iterations);
    
    let par_time = benchmark_function("  Parallel CCE Derivative", || {
        let _ = pred_par.categorical_cross_entropy_derivative(&target_par);
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    // Test 21: Argmax
    println!("21. ARGMAX");
    let seq_time = benchmark_function("  Sequential Argmax", || {
        let _ = col_seq.argmax();
    }, iterations);
    
    let par_time = benchmark_function("  Parallel Argmax", || {
        let _ = col_par.argmax();
    }, iterations);
    
    println!("  Speedup: {:.2}x\n", seq_time / par_time);
    
    println!("=== Benchmark Complete ===");
    println!("Note: Speedup values > 1.0 indicate parallel is faster");
    println!("Values < 1.0 indicate sequential is faster (overhead cost)");
    
    // System info
    println!("\nSystem Info:");
    println!("Available parallelism: {} threads", 
        std::thread::available_parallelism().unwrap().get());
}

// Additional benchmarking functions for specific scenarios
#[allow(dead_code)]
fn benchmark_different_sizes() {
    println!("\n=== Size-based Performance Analysis ===");
    let sizes = vec![50, 100, 200, 500, 1000];
    
    for &size in &sizes {
        println!("\nMatrix size: {}x{}", size, size);
        
        let a_seq = Tensor::new_with_concurrent(
            (0..size * size).map(|i| i as f64).collect(),
            size, size, false
        );
        let b_seq = Tensor::new_with_concurrent(
            (0..size * size).map(|i| (i * 2) as f64).collect(),
            size, size, false
        );
        let a_par = Tensor::new_with_concurrent(
            (0..size * size).map(|i| i as f64).collect(),
            size, size, true
        );
        let b_par = Tensor::new_with_concurrent(
            (0..size * size).map(|i| (i * 2) as f64).collect(),
            size, size, true
        );
        
        let iterations = if size > 500 { 10 } else { 50 };
        
        let seq_time = benchmark_function("  Sequential Add", || {
            let _ = &a_seq + &b_seq;
        }, iterations);
        
        let par_time = benchmark_function("  Parallel Add", || {
            let _ = &a_par + &b_par;
        }, iterations);
        
        println!("  Speedup: {:.2}x", seq_time / par_time);
    }
}

#[allow(dead_code)]
fn benchmark_matrix_multiplication_sizes() {
    println!("\n=== Matrix Multiplication Size Analysis ===");
    let sizes = vec![50, 100, 200, 300];
    
    for &size in &sizes {
        println!("\nMatrix multiplication: {}x{} * {}x{}", size, size, size, size);
        
        let m1_seq = Tensor::new_with_concurrent(
            (0..size * size).map(|i| (i as f64) * 0.01).collect(),
            size, size, false
        );
        let m2_seq = Tensor::new_with_concurrent(
            (0..size * size).map(|i| (i as f64) * 0.02).collect(),
            size, size, false
        );
        let m1_par = Tensor::new_with_concurrent(
            (0..size * size).map(|i| (i as f64) * 0.01).collect(),
            size, size, true
        );
        let m2_par = Tensor::new_with_concurrent(
            (0..size * size).map(|i| (i as f64) * 0.02).collect(),
            size, size, true
        );
        
        let iterations = if size > 200 { 5 } else { 20 };
        
        let seq_time = benchmark_function("  Sequential Mul", || {
            let _ = m1_seq.mul(&m2_seq);
        }, iterations);
        
        let par_time = benchmark_function("  Parallel Mul", || {
            let _ = m1_par.mul(&m2_par);
        }, iterations);
        
        println!("  Speedup: {:.2}x", seq_time / par_time);
    }
}