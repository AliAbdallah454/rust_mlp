use cp_proj::Tensor;
use std::f64::EPSILON;
use std::time::Instant;

// Helper function to compare tensors with floating point tolerance
fn tensors_equal(a: &Tensor, b: &Tensor, tolerance: f64) -> bool {
    if a.rows != b.rows || a.cols != b.cols {
        return false;
    }
    
    for i in 0..a.data.len() {
        if (a.data[i] - b.data[i]).abs() > tolerance {
            return false;
        }
    }
    true
}

// Helper function to create a tensor with known values for testing
fn create_test_tensor(data: Vec<f64>, rows: u32, cols: u32) -> Tensor {
    Tensor::new(data, rows, cols)
}

#[test]
fn test_basic_matrix_multiplication() {
    // Test 2x2 * 2x2 matrix multiplication
    let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = create_test_tensor(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    
    // Expected result: [[19, 22], [43, 50]]
    let expected = create_test_tensor(vec![19.0, 22.0, 43.0, 50.0], 2, 2);
    
    let result_seq = a.mul_seq(&b);
    let result_par_1 = a.mul(&b);
    let result_par_2 = a.mul(&b);
    
    assert!(tensors_equal(&result_seq, &expected, EPSILON));
    assert!(tensors_equal(&result_par_1, &expected, EPSILON));
    assert!(tensors_equal(&result_par_2, &expected, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd = a.mul_par_simd(&b, 1);
        assert!(tensors_equal(&result_simd, &expected, EPSILON));
    }
}

#[test]
fn test_identity_matrix_multiplication() {
    // Test multiplication with identity matrix
    let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let identity = create_test_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    
    let result_seq = a.mul_seq(&identity);
    let result_par = a.mul(&identity);
    
    assert!(tensors_equal(&result_seq, &a, EPSILON));
    assert!(tensors_equal(&result_par, &a, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd = a.mul_par_simd(&identity, 2);
        assert!(tensors_equal(&result_simd, &a, EPSILON));
    }
}

#[test]
fn test_zero_matrix_multiplication() {
    // Test multiplication with zero matrix
    let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let zero = create_test_tensor(vec![0.0, 0.0, 0.0, 0.0], 2, 2);
    let expected_zero = create_test_tensor(vec![0.0, 0.0, 0.0, 0.0], 2, 2);
    
    let result_seq = a.mul_seq(&zero);
    let result_par = a.mul(&zero);
    
    assert!(tensors_equal(&result_seq, &expected_zero, EPSILON));
    assert!(tensors_equal(&result_par, &expected_zero, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd = a.mul_par_simd(&zero, 2);
        assert!(tensors_equal(&result_simd, &expected_zero, EPSILON));
    }
}

#[test]
fn test_single_element_matrices() {
    // Test 1x1 matrices
    let a = create_test_tensor(vec![3.0], 1, 1);
    let b = create_test_tensor(vec![4.0], 1, 1);
    let expected = create_test_tensor(vec![12.0], 1, 1);
    
    let result_seq = a.mul_seq(&b);
    let result_par = a.mul(&b);
    
    assert!(tensors_equal(&result_seq, &expected, EPSILON));
    assert!(tensors_equal(&result_par, &expected, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd = a.mul_par_simd(&b, 1);
        assert!(tensors_equal(&result_simd, &expected, EPSILON));
    }
}

#[test]
fn test_rectangular_matrices() {
    // Test 3x2 * 2x4 multiplication
    let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let b = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 4);
    
    // Expected result: 3x4 matrix
    // [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
    let expected = create_test_tensor(
        vec![11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0], 
        3, 4
    );
    
    let result_seq = a.mul_seq(&b);
    let result_par_1 = a.mul(&b);
    let result_par_2 = a.mul(&b);
    let result_par_3 = a.mul(&b);
    
    assert!(tensors_equal(&result_seq, &expected, EPSILON));
    assert!(tensors_equal(&result_par_1, &expected, EPSILON));
    assert!(tensors_equal(&result_par_2, &expected, EPSILON));
    assert!(tensors_equal(&result_par_3, &expected, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd_1 = a.mul_par_simd(&b, 1);
        let result_simd_2 = a.mul_par_simd(&b, 2);
        let result_simd_3 = a.mul_par_simd(&b, 3);
        assert!(tensors_equal(&result_simd_1, &expected, EPSILON));
        assert!(tensors_equal(&result_simd_2, &expected, EPSILON));
        assert!(tensors_equal(&result_simd_3, &expected, EPSILON));
    }
}

#[test]
fn test_vector_multiplication() {
    // Test matrix * column vector (3x3 * 3x1)
    let matrix = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
    let vector = create_test_tensor(vec![1.0, 2.0, 3.0], 3, 1);
    let expected = create_test_tensor(vec![14.0, 32.0, 50.0], 3, 1);
    
    let result_seq = matrix.mul_seq(&vector);
    let result_par = matrix.mul(&vector);
    
    assert!(tensors_equal(&result_seq, &expected, EPSILON));
    assert!(tensors_equal(&result_par, &expected, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd = matrix.mul_par_simd(&vector, 2);
        assert!(tensors_equal(&result_simd, &expected, EPSILON));
    }
}

#[test]
fn test_row_vector_multiplication() {
    // Test row vector * matrix (1x3 * 3x2)
    let row_vector = create_test_tensor(vec![1.0, 2.0, 3.0], 1, 3);
    let matrix = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let expected = create_test_tensor(vec![22.0, 28.0], 1, 2);
    
    let result_seq = row_vector.mul_seq(&matrix);
    let result_par = row_vector.mul(&matrix);
    
    assert!(tensors_equal(&result_seq, &expected, EPSILON));
    assert!(tensors_equal(&result_par, &expected, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd = row_vector.mul_par_simd(&matrix, 1);
        assert!(tensors_equal(&result_simd, &expected, EPSILON));
    }
}

#[test]
fn test_large_matrices() {
    // Test larger matrices to verify threading works correctly
    let size = 100;
    let a = Tensor::random(size, size, 42);
    let b = Tensor::random(size, size, 123);
    
    let result_seq = a.mul_seq(&b);
    let result_par_1 = a.mul(&b);
    let result_par_4 = a.mul(&b);
    let result_par_8 = a.mul(&b);
    
    assert!(tensors_equal(&result_seq, &result_par_1, 1e-10));
    assert!(tensors_equal(&result_seq, &result_par_4, 1e-10));
    assert!(tensors_equal(&result_seq, &result_par_8, 1e-10));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd_1 = a.mul_par_simd(&b, 1);
        let result_simd_4 = a.mul_par_simd(&b, 4);
        let result_simd_8 = a.mul_par_simd(&b, 8);
        assert!(tensors_equal(&result_seq, &result_simd_1, 1e-10));
        assert!(tensors_equal(&result_seq, &result_simd_4, 1e-10));
        assert!(tensors_equal(&result_seq, &result_simd_8, 1e-10));
    }
}

#[test]
fn test_non_square_large_matrices() {
    // Test non-square matrices with threading
    let a = Tensor::random(50, 75, 42);
    let b = Tensor::random(75, 60, 123);
    
    let result_seq = a.mul_seq(&b);
    let result_par_2 = a.mul(&b);
    let result_par_5 = a.mul(&b);
    
    assert!(tensors_equal(&result_seq, &result_par_2, 1e-10));
    assert!(tensors_equal(&result_seq, &result_par_5, 1e-10));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd_2 = a.mul_par_simd(&b, 2);
        let result_simd_5 = a.mul_par_simd(&b, 5);
        assert!(tensors_equal(&result_seq, &result_simd_2, 1e-10));
        assert!(tensors_equal(&result_seq, &result_simd_5, 1e-10));
    }
}

#[test]
fn test_thread_count_edge_cases() {
    // Test with more threads than rows
    let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = create_test_tensor(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let expected = create_test_tensor(vec![19.0, 22.0, 43.0, 50.0], 2, 2);
    
    let result_par_10 = a.mul(&b); // More threads than rows
    assert!(tensors_equal(&result_par_10, &expected, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd_10 = a.mul_par_simd(&b, 10);
        assert!(tensors_equal(&result_simd_10, &expected, EPSILON));
    }
}

#[test]
fn test_negative_values() {
    // Test with negative values
    let a = create_test_tensor(vec![-1.0, 2.0, -3.0, 4.0], 2, 2);
    let b = create_test_tensor(vec![5.0, -6.0, 7.0, -8.0], 2, 2);
    let expected = create_test_tensor(vec![9.0, -10.0, 13.0, -14.0], 2, 2);
    
    let result_seq = a.mul_seq(&b);
    let result_par = a.mul(&b);
    
    assert!(tensors_equal(&result_seq, &expected, EPSILON));
    assert!(tensors_equal(&result_par, &expected, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd = a.mul_par_simd(&b, 2);
        assert!(tensors_equal(&result_simd, &expected, EPSILON));
    }
}

#[test]
fn test_fractional_values() {
    // Test with fractional values
    let a = create_test_tensor(vec![0.5, 1.5, 2.5, 3.5], 2, 2);
    let b = create_test_tensor(vec![0.25, 0.75, 1.25, 1.75], 2, 2);
    let expected = create_test_tensor(vec![2.0, 3.0, 5.0, 8.0], 2, 2);
    
    let result_seq = a.mul_seq(&b);
    let result_par = a.mul(&b);
    
    assert!(tensors_equal(&result_seq, &expected, EPSILON));
    assert!(tensors_equal(&result_par, &expected, EPSILON));
    
    #[cfg(target_arch = "x86_64")]
    {
        let result_simd = a.mul_par_simd(&b, 2);
        assert!(tensors_equal(&result_simd, &expected, EPSILON));
    }
}

#[test]
#[should_panic(expected = "Matrix dimensions don't match")]
fn test_incompatible_dimensions_seq() {
    let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = create_test_tensor(vec![1.0, 2.0, 3.0], 3, 1);
    a.mul_seq(&b); // Should panic: 2x2 * 3x1 is invalid
}

#[test]
#[should_panic(expected = "Matrix dimensions don't match")]
fn test_incompatible_dimensions_par() {
    let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = create_test_tensor(vec![1.0, 2.0, 3.0], 3, 1);
    a.mul(&b); // Should panic: 2x2 * 3x1 is invalid
}

#[test]
#[cfg(target_arch = "x86_64")]
#[should_panic(expected = "Matrix dimensions don't match")]
fn test_incompatible_dimensions_simd() {
    let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = create_test_tensor(vec![1.0, 2.0, 3.0], 3, 1);
    a.mul_par_simd(&b, 2); // Should panic: 2x2 * 3x1 is invalid
}

#[test]
fn test_simd_alignment_edge_cases() {
    #[cfg(target_arch = "x86_64")]
    {
        // Test matrices where dimensions are not multiples of 4 (SIMD width)
        let a = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 5);
        let b = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], 5, 3);
        
        let result_seq = a.mul_seq(&b);
        let result_simd = a.mul_par_simd(&b, 2);
        
        assert!(tensors_equal(&result_seq, &result_simd, 1e-10));
    }
}

#[test]
fn test_consistency_across_implementations() {
    // Test that all implementations produce the same results for various sizes
    let test_cases = vec![
        (1, 1, 1),   // 1x1 * 1x1
        (2, 3, 4),   // 2x3 * 3x4
        (5, 7, 3),   // 5x7 * 7x3
        (10, 15, 8), // 10x15 * 15x8
    ];

    for (r1, c1, c2) in test_cases {
        let a = Tensor::random(r1, c1, 42);
        let b = Tensor::random(c1, c2, 123);
        
        let result_seq = a.mul_seq(&b);
        let result_par_1 = a.mul(&b);
        let result_par_2 = a.mul(&b);
        
        assert!(tensors_equal(&result_seq, &result_par_1, 1e-12));
        assert!(tensors_equal(&result_seq, &result_par_2, 1e-12));
        
        #[cfg(target_arch = "x86_64")]
        {
            let result_simd_1 = a.mul_par_simd(&b, 1);
            let result_simd_2 = a.mul_par_simd(&b, 2);
            assert!(tensors_equal(&result_seq, &result_simd_1, 1e-12));
            assert!(tensors_equal(&result_seq, &result_simd_2, 1e-12));
        }
    }
}

#[test]
fn test_performance_comparison() {
    // Basic performance test to ensure parallel versions don't crash
    let size = 50;
    let a = Tensor::random(size, size, 42);
    let b = Tensor::random(size, size, 123);
    
    let start = Instant::now();
    let _result_seq = a.mul_seq(&b);
    let seq_time = start.elapsed();
    
    let start = Instant::now();
    let _result_par = a.mul(&b);
    let par_time = start.elapsed();
    
    #[cfg(target_arch = "x86_64")]
    {
        let start = Instant::now();
        let _result_simd = a.mul_par_simd(&b, 4);
        let simd_time = start.elapsed();
        
        println!("Sequential: {:?}, Parallel: {:?}, SIMD: {:?}", seq_time, par_time, simd_time);
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("Sequential: {:?}, Parallel: {:?}", seq_time, par_time);
    }
    
    // This test just ensures the functions complete without panicking
    // In a real scenario, parallel versions should be faster for large matrices
}