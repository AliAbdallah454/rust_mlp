// use cp_proj::tensor::{Tensor, ExecutionMode};
// use std::f32::EPSILON;
// use std::time::Instant;

// #[cfg(test)]
// mod tensor_tests {
//     use super::*;

//     // Helper function to compare tensors with floating point tolerance
//     fn tensors_equal(a: &Tensor, b: &Tensor, tolerance: f32) -> bool {
//         if a.rows != b.rows || a.cols != b.cols {
//             return false;
//         }
        
//         a.data.iter()
//             .zip(b.data.iter())
//             .all(|(x, y)| (x - y).abs() < tolerance)
//     }

//     // Helper function to create a tensor with known values for testing
//     fn create_test_tensor(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
//         Tensor::new(data, rows, cols)
//     }

//     // CONSTRUCTOR TESTS
//     #[test]
//     fn test_new_constructor() {
//         let data = vec![1.0, 2.0, 3.0, 4.0];
//         let tensor = Tensor::new(data.clone(), 2, 2);
        
//         assert_eq!(tensor.rows, 2);
//         assert_eq!(tensor.cols, 2);
//         assert_eq!(tensor.data, data);
//     }

//     #[test]
//     fn test_scalar_constructor() {
//         let tensor = Tensor::scalar(5.0);
        
//         assert_eq!(tensor.rows, 1);
//         assert_eq!(tensor.cols, 1);
//         assert_eq!(tensor.data, vec![5.0]);
//     }

//     #[test]
//     fn test_random_constructor() {
//         let rows = 3;
//         let cols = 4;
//         let tensor = Tensor::random(rows, cols, 42);
        
//         assert_eq!(tensor.rows, rows);
//         assert_eq!(tensor.cols, cols);
//         assert_eq!(tensor.data.len(), rows * cols);
        
//         // Check that values are in the expected range [0.0, 1.0]
//         for val in &tensor.data {
//             assert!(*val >= 0.0 && *val <= 1.0);
//         }
        
//         // Check that random with same seed produces same values
//         let tensor2 = Tensor::random(rows, cols, 42);
//         assert_eq!(tensor.data, tensor2.data);
        
//         // Check that random with different seed produces different values
//         let tensor3 = Tensor::random(rows, cols, 43);
//         assert_ne!(tensor.data, tensor3.data);
//     }

//     #[test]
//     fn test_ones_constructor() {
//         let rows = 2;
//         let cols = 3;
//         let tensor = Tensor::ones(rows, cols);
        
//         assert_eq!(tensor.rows, rows);
//         assert_eq!(tensor.cols, cols);
//         assert_eq!(tensor.data, vec![1.0; rows * cols]);
//     }

//     #[test]
//     fn test_zeros_constructor() {
//         let rows = 3;
//         let cols = 2;
//         let tensor = Tensor::zeros(rows, cols);
        
//         assert_eq!(tensor.rows, rows);
//         assert_eq!(tensor.cols, cols);
//         assert_eq!(tensor.data, vec![0.0; rows * cols]);
//     }

//     // BASIC OPERATIONS TESTS
//     #[test]
//     fn test_dims() {
//         let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
//         let (rows, cols) = tensor.dims();
        
//         assert_eq!(rows, 2);
//         assert_eq!(cols, 3);
//     }

//     #[test]
//     fn test_transpose() {
//         let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
//         let transposed = tensor.transpose();
        
//         assert_eq!(transposed.rows, 3);
//         assert_eq!(transposed.cols, 2);
//         assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
//     }

//     #[test]
//     fn test_scale() {
//         let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let scaled = tensor.scale(2.0);
        
//         assert_eq!(scaled.rows, 2);
//         assert_eq!(scaled.cols, 2);
//         assert_eq!(scaled.data, vec![2.0, 4.0, 6.0, 8.0]);
//     }

//     #[test]
//     fn test_sum() {
//         let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let sum = tensor.sum();
        
//         assert_eq!(sum, 10.0);
//     }

//     #[test]
//     fn test_square() {
//         let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let squared = tensor.square();
        
//         assert_eq!(squared.rows, 2);
//         assert_eq!(squared.cols, 2);
//         assert_eq!(squared.data, vec![1.0, 4.0, 9.0, 16.0]);
//     }

//     #[test]
//     fn test_argmax() {
//         let tensor = Tensor::new(vec![3.0, 1.0, 4.0, 2.0], 4, 1);
//         let max_idx = tensor.argmax();
        
//         assert_eq!(max_idx, 2); // Index of 4.0
//     }

//     // OPERATOR TESTS
//     #[test]
//     fn test_add_operator() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let result = &a + &b;
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
//     }

//     #[test]
//     fn test_sub_operator_refs() {
//         let a = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let result = &a - &b;
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![4.0, 4.0, 4.0, 4.0]);
//     }

//     #[test]
//     fn test_sub_operator_owned() {
//         let a = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let result = a - b;
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![4.0, 4.0, 4.0, 4.0]);
//     }

//     #[test]
//     fn test_partial_eq() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let c = Tensor::new(vec![1.0, 2.0, 3.0, 4.1], 2, 2);
//         let d = Tensor::new(vec![1.0, 2.0, 3.0], 3, 1);
        
//         assert_eq!(a, b);
//         assert_ne!(a, c);
//         assert_ne!(a, d);
        
//         // Test with small differences (within epsilon)
//         let almost_a = Tensor::new(vec![1.001, 2.002, 3.003, 4.004], 2, 2);
//         assert_eq!(a, almost_a); // Should be equal due to epsilon comparison
//     }

//     // HADAMARD PRODUCT TESTS
//     #[test]
//     fn test_hadamard() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let result = a.hadamard(&b);
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![5.0, 12.0, 21.0, 32.0]);
//     }

//     // ACTIVATION FUNCTION TESTS
//     #[test]
//     fn test_relu() {
//         let tensor = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], 2, 2);
//         let result = tensor.relu();
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![0.0, 0.0, 1.0, 2.0]);
//     }

//     #[test]
//     fn test_relu_derivative() {
//         let tensor = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], 2, 2);
//         let result = tensor.relu_derivative();
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![0.0, 0.0, 1.0, 1.0]);
//     }

//     #[test]
//     fn test_sigmoid() {
//         let tensor = Tensor::new(vec![-2.0, 0.0, 2.0], 3, 1);
//         let result = tensor.sigmoid();
        
//         assert_eq!(result.rows, 3);
//         assert_eq!(result.cols, 1);
        
//         // Expected values: sigmoid(-2) ≈ 0.119, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.881
//         assert!((result.data[0] - 0.119).abs() < 0.001);
//         assert!((result.data[1] - 0.5).abs() < 0.001);
//         assert!((result.data[2] - 0.881).abs() < 0.001);
//     }

//     #[test]
//     fn test_sigmoid_derivative() {
//         let tensor = Tensor::new(vec![-2.0, 0.0, 2.0], 3, 1);
//         let result = tensor.sigmoid_derivative();
        
//         assert_eq!(result.rows, 3);
//         assert_eq!(result.cols, 1);
        
//         // Expected values: sigmoid(-2)*(1-sigmoid(-2)) ≈ 0.105, sigmoid(0)*(1-sigmoid(0)) = 0.25, sigmoid(2)*(1-sigmoid(2)) ≈ 0.105
//         assert!((result.data[0] - 0.105).abs() < 0.001);
//         assert!((result.data[1] - 0.25).abs() < 0.001);
//         assert!((result.data[2] - 0.105).abs() < 0.001);
//     }

//     #[test]
//     fn test_tanh() {
//         let tensor = Tensor::new(vec![-2.0, 0.0, 2.0], 3, 1);
//         let result = tensor.tanh();
        
//         assert_eq!(result.rows, 3);
//         assert_eq!(result.cols, 1);
        
//         // Expected values: tanh(-2) ≈ -0.964, tanh(0) = 0, tanh(2) ≈ 0.964
//         assert!((result.data[0] - (-0.964)).abs() < 0.001);
//         assert!((result.data[1] - 0.0).abs() < 0.001);
//         assert!((result.data[2] - 0.964).abs() < 0.001);
//     }

//     #[test]
//     fn test_tanh_derivative() {
//         let tensor = Tensor::new(vec![-2.0, 0.0, 2.0], 3, 1);
//         let result = tensor.tanh_derivative();
        
//         assert_eq!(result.rows, 3);
//         assert_eq!(result.cols, 1);
        
//         // Expected values: 1 - tanh(-2)^2 ≈ 0.07, 1 - tanh(0)^2 = 1, 1 - tanh(2)^2 ≈ 0.07
//         assert!((result.data[0] - 0.07).abs() < 0.001);
//         assert!((result.data[1] - 1.0).abs() < 0.001);
//         assert!((result.data[2] - 0.07).abs() < 0.001);
//     }

//     #[test]
//     fn test_softmax() {
//         let tensor = Tensor::new(vec![1.0, 2.0, 3.0], 3, 1);
//         let result = tensor.softmax();
        
//         assert_eq!(result.rows, 3);
//         assert_eq!(result.cols, 1);
        
//         // Expected values: e^1/(e^1+e^2+e^3) ≈ 0.09, e^2/(e^1+e^2+e^3) ≈ 0.24, e^3/(e^1+e^2+e^3) ≈ 0.67
//         assert!((result.data[0] - 0.09).abs() < 0.01);
//         assert!((result.data[1] - 0.24).abs() < 0.01);
//         assert!((result.data[2] - 0.67).abs() < 0.01);
        
//         // Sum should be 1.0
//         assert!((result.data.iter().sum::<f32>() - 1.0).abs() < 0.001);
//     }

//     #[test]
//     fn test_softmax_derivative() {
//         let tensor = Tensor::new(vec![1.0, 2.0], 2, 1);
//         let result = tensor.softmax_derivative();
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
        
//         // For a 2x1 vector, the Jacobian is 2x2
//         // Verify it's a valid Jacobian matrix
//         let softmax_vals = tensor.softmax();
//         let s0 = softmax_vals.data[0];
//         let s1 = softmax_vals.data[1];
        
//         // Expected Jacobian: [[s0*(1-s0), -s0*s1], [-s1*s0, s1*(1-s1)]]
//         assert!((result.data[0] - s0*(1.0-s0)).abs() < 0.001); // J[0,0]
//         assert!((result.data[1] - (-s0*s1)).abs() < 0.001);     // J[0,1]
//         assert!((result.data[2] - (-s1*s0)).abs() < 0.001);     // J[1,0]
//         assert!((result.data[3] - s1*(1.0-s1)).abs() < 0.001); // J[1,1]
//     }

//     // LOSS FUNCTION TESTS
//     #[test]
//     fn test_mse_loss() {
//         let predictions = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let targets = Tensor::new(vec![1.5, 2.5, 3.5, 4.5], 2, 2);
//         let loss = predictions.mse_loss(&targets);
        
//         // Expected: ((1.0-1.5)^2 + (2.0-2.5)^2 + (3.0-3.5)^2 + (4.0-4.5)^2) / 4 = 0.25
//         assert!((loss - 0.25).abs() < 0.001);
//     }

//     #[test]
//     fn test_mse_loss_derivative() {
//         let predictions = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let targets = Tensor::new(vec![1.5, 2.5, 3.5, 4.5], 2, 2);
//         let derivative = predictions.mse_loss_derivative(&targets);
        
//         assert_eq!(derivative.rows, 2);
//         assert_eq!(derivative.cols, 2);
        
//         // Expected: 2*(predictions - targets)/4 = (predictions - targets)/2
//         let expected = Tensor::new(vec![-0.25, -0.25, -0.25, -0.25], 2, 2);
//         assert!(tensors_equal(&derivative, &expected, 0.001));
//     }

//     #[test]
//     fn test_categorical_cross_entropy() {
//         let predictions = Tensor::new(vec![0.7, 0.2, 0.1], 3, 1);
//         let targets = Tensor::new(vec![1.0, 0.0, 0.0], 3, 1);
//         let loss = predictions.categorical_cross_entropy(&targets);
        
//         // Expected: -1.0*ln(0.7) ≈ 0.357
//         assert!((loss - 0.357).abs() < 0.001);
//     }

//     #[test]
//     fn test_categorical_cross_entropy_derivative() {
//         let predictions = Tensor::new(vec![0.7, 0.2, 0.1], 3, 1);
//         let targets = Tensor::new(vec![1.0, 0.0, 0.0], 3, 1);
//         let derivative = predictions.categorical_cross_entropy_derivative(&targets);
        
//         assert_eq!(derivative.rows, 3);
//         assert_eq!(derivative.cols, 1);
        
//         // Expected: (predictions - targets)/3 = [-0.1, 0.067, 0.033]
//         let expected = Tensor::new(vec![-0.1, 0.067, 0.033], 3, 1);
//         assert!(tensors_equal(&derivative, &expected, 0.001));
//     }

//     // MATRIX MULTIPLICATION TESTS
//     #[test]
//     fn test_mul_vec() {
//         let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let vector = Tensor::new(vec![5.0, 6.0], 2, 1);
//         let result = matrix.mul_vec(&vector);
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 1);
//         assert_eq!(result.data, vec![17.0, 39.0]);
//     }

//     #[test]
//     fn test_mul_vec_parallel() {
//         let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let vector = Tensor::new(vec![5.0, 6.0], 2, 1);
//         let result = matrix.mul_vec_parallel(&vector, 2);
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 1);
//         assert_eq!(result.data, vec![17.0, 39.0]);
//     }

//     #[test]
//     fn test_mul_simd() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let result = a.mul_simd(&b);
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
//     }

//     #[test]
//     fn test_mul_simd_parallel() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let result = a.mul_simd_parallel(&b, 2);
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
//     }

//     #[test]
//     fn test_mul_seq() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let result = a.mul_seq(&b);
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
//     }

//     #[test]
//     fn test_mul_par() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let result = a.mul_par(&b, 2);
        
//         assert_eq!(result.rows, 2);
//         assert_eq!(result.cols, 2);
//         assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
//     }

//     #[test]
//     fn test_mul_with_execution_mode() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
//         let expected = Tensor::new(vec![19.0, 22.0, 43.0, 50.0], 2, 2);
        
//         let result_seq = a.mul(&b, ExecutionMode::Sequential);
//         let result_par = a.mul(&b, ExecutionMode::Parallel);
//         let result_simd = a.mul(&b, ExecutionMode::SIMD);
//         let result_par_simd = a.mul(&b, ExecutionMode::ParallelSIMD);
        
//         assert_eq!(result_seq, expected);
//         assert_eq!(result_par, expected);
//         assert_eq!(result_simd, expected);
//         assert_eq!(result_par_simd, expected);
//     }

//     // EDGE CASES AND ERROR HANDLING TESTS
//     #[test]
//     #[should_panic(expected = "Tensor add: row mismatch")]
//     fn test_add_dimension_mismatch_rows() {
//         let a = Tensor::new(vec![1.0, 2.0], 1, 2);
//         let b = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], 2, 2);
//         let _ = &a + &b; // Should panic
//     }

//     #[test]
//     #[should_panic(expected = "Tensor add: col mismatch")]
//     fn test_add_dimension_mismatch_cols() {
//         let a = Tensor::new(vec![1.0, 2.0], 1, 2);
//         let b = Tensor::new(vec![3.0], 1, 1);
//         let _ = &a + &b; // Should panic
//     }

//     #[test]
//     #[should_panic(expected = "Expected vector to be rx1")]
//     fn test_mul_vec_not_column_vector() {
//         let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let not_vector = Tensor::new(vec![5.0, 6.0], 1, 2); // Row vector, not column
//         matrix.mul_vec(&not_vector); // Should panic
//     }

//     #[test]
//     #[should_panic(expected = "Expected tensor dimensions to match")]
//     fn test_mul_simd_dimension_mismatch() {
//         let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         let b = Tensor::new(vec![5.0, 6.0, 7.0], 3, 1);
//         a.mul_simd(&b); // Should panic
//     }

//     #[test]
//     #[should_panic(expected = "argmax only works on column vectors")]
//     fn test_argmax_not_column_vector() {
//         let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         tensor.argmax(); // Should panic
//     }

//     #[test]
//     #[should_panic(expected = "Softmax only implemented for column vectors")]
//     fn test_softmax_not_column_vector() {
//         let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//         tensor.softmax(); // Should panic
//     }

//     // PERFORMANCE TESTS
//     #[test]
//     fn test_performance_comparison() {
//         // This test just ensures all multiplication methods complete without panicking
//         // and provides timing information
//         let size = 50;
//         let a = Tensor::random(size, size, 42);
//         let b = Tensor::random(size, size, 123);
        
//         let start = Instant::now();
//         let result_seq = a.mul_seq(&b);
//         let seq_time = start.elapsed();
        
//         let start = Instant::now();
//         let result_par = a.mul_par(&b, 4);
//         let par_time = start.elapsed();
        
//         let start = Instant::now();
//         let result_simd = a.mul_simd(&b);
//         let simd_time = start.elapsed();
        
//         let start = Instant::now();
//         let result_par_simd = a.mul_simd_parallel(&b, 4);
//         let par_simd_time = start.elapsed();
        
//         println!("Sequential: {:?}, Parallel: {:?}, SIMD: {:?}, Parallel SIMD: {:?}", 
//                  seq_time, par_time, simd_time, par_simd_time);
        
//         // Verify all implementations produce the same result
//         assert!(tensors_equal(&result_seq, &result_par, 1e-5));
//         assert!(tensors_equal(&result_seq, &result_simd, 1e-5));
//         assert!(tensors_equal(&result_seq, &result_par_simd, 1e-5));
//     }
// }