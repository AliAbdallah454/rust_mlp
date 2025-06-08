use cp_proj::tensor::Tensor;

#[cfg(test)]
mod hadamard_tests {
    use super::*;

    // Helper function to compare tensors with floating point tolerance
    fn tensors_equal(a: &Tensor, b: &Tensor, tolerance: f32) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        
        a.data.iter()
            .zip(b.data.iter())
            .all(|(x, y)| (x - y).abs() < tolerance)
    }

    #[test]
    fn test_hadamard_basic_2x2() {
        // Test basic 2x2 matrices
        let a = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], 2, 2);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        
        let result = a.hadamard(&b);
        let expected = Tensor::new(vec![2.0, 6.0, 12.0, 20.0], 2, 2);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
    }

    #[test]
    fn test_hadamard_1x1() {
        // Test single element
        let a = Tensor::new(vec![5.0], 1, 1);
        let b = Tensor::new(vec![3.0], 1, 1);
        
        let result = a.hadamard(&b);
        let expected = Tensor::new(vec![15.0], 1, 1);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
    }

    #[test]
    fn test_hadamard_row_vector() {
        // Test 1x4 row vector
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 1, 4);
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], 1, 4);
        
        let result = a.hadamard(&b);
        let expected = Tensor::new(vec![2.0, 6.0, 12.0, 20.0], 1, 4);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
        assert_eq!(result.rows, 1);
        assert_eq!(result.cols, 4);
    }

    #[test]
    fn test_hadamard_column_vector() {
        // Test 4x1 column vector
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], 4, 1);
        
        let result = a.hadamard(&b);
        let expected = Tensor::new(vec![2.0, 6.0, 12.0, 20.0], 4, 1);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
        assert_eq!(result.rows, 4);
        assert_eq!(result.cols, 1);
    }

    #[test]
    fn test_hadamard_with_zeros() {
        // Test with zeros
        let a = Tensor::new(vec![1.0, 0.0, 3.0, 0.0], 2, 2);
        let b = Tensor::new(vec![2.0, 5.0, 0.0, 4.0], 2, 2);
        
        let result = a.hadamard(&b);
        let expected = Tensor::new(vec![2.0, 0.0, 0.0, 0.0], 2, 2);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
    }

    #[test]
    fn test_hadamard_with_negatives() {
        // Test with negative numbers
        let a = Tensor::new(vec![-2.0, 3.0, -4.0, 5.0], 2, 2);
        let b = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], 2, 2);
        
        let result = a.hadamard(&b);
        let expected = Tensor::new(vec![-2.0, -6.0, -12.0, -20.0], 2, 2);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
    }

    #[test]
    fn test_hadamard_with_decimals() {
        // Test with floating point numbers
        let a = Tensor::new(vec![1.5, 2.7, 3.14, 0.5], 2, 2);
        let b = Tensor::new(vec![2.0, 1.1, 2.0, 4.0], 2, 2);
        
        let result = a.hadamard(&b);
        let expected = Tensor::new(vec![3.0, 2.97, 6.28, 2.0], 2, 2);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
    }

    #[test]
    fn test_hadamard_large_matrix() {
        // Test with larger matrix (8x8 = 64 elements) to test threading
        let size = 8;
        let a_data: Vec<f32> = (1..=size*size).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (1..=size*size).map(|x| (x * 2) as f32).collect();
        
        let a = Tensor::new(a_data.clone(), size, size);
        let b = Tensor::new(b_data.clone(), size, size);
        
        let result = a.hadamard(&b);
        
        // Expected: each element should be a[i] * b[i] = i * (i * 2) = 2 * i^2
        let expected_data: Vec<f32> = (1..=size*size).map(|x| (x * x * 2) as f32).collect();
        let expected = Tensor::new(expected_data, size, size);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
        assert_eq!(result.rows, size);
        assert_eq!(result.cols, size);
    }

    #[test]
    fn test_hadamard_very_large_matrix() {
        // Test with matrix large enough to ensure multiple threads are used
        let rows = 100;
        let cols = 50;
        let total_elements = rows * cols;
        
        let a_data: Vec<f32> = (0..total_elements).map(|x| (x as f32) + 1.0).collect();
        let b_data: Vec<f32> = (0..total_elements).map(|x| (x as f32) * 2.0 + 1.0).collect();
        
        let a = Tensor::new(a_data.clone(), rows, cols);
        let b = Tensor::new(b_data.clone(), rows, cols);
        
        let result = a.hadamard(&b);
        
        // Verify dimensions
        assert_eq!(result.rows, rows);
        assert_eq!(result.cols, cols);
        
        // Verify each element
        for i in 0..total_elements {
            let expected_val = a_data[i as usize] * b_data[i as usize];
            assert!((result.data[i as usize] - expected_val).abs() < 1e-10,
                    "Mismatch at index {}: got {}, expected {}", 
                    i, result.data[i as usize], expected_val);
        }
    }

    #[test]
    fn test_hadamard_identity_property() {
        // Test A ⊙ I = A where I is tensor of ones
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let ones = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], 2, 2);
        
        let result = a.hadamard(&ones);
        
        assert!(tensors_equal(&result, &a, 1e-10));
    }

    #[test]
    fn test_hadamard_commutative_property() {
        // Test A ⊙ B = B ⊙ A (commutative property)
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        
        let ab = a.hadamard(&b);
        let ba = b.hadamard(&a);
        
        assert!(tensors_equal(&ab, &ba, 1e-10));
    }

    #[test]
    #[should_panic(expected = "Tensor hadamard: row mismatch")]
    fn test_hadamard_dimension_mismatch_rows() {
        let a = Tensor::new(vec![1.0, 2.0], 1, 2);  // 1x2
        let b = Tensor::new(vec![1.0, 2.0], 2, 1);  // 2x1
        
        a.hadamard(&b); // Should panic
    }

    #[test]
    #[should_panic(expected = "Tensor hadamard: col mismatch")]
    fn test_hadamard_dimension_mismatch_cols() {
        let a = Tensor::new(vec![1.0, 2.0], 1, 2);  // 1x2
        let b = Tensor::new(vec![1.0, 2.0, 3.0], 1, 3);  // 1x3
        
        a.hadamard(&b); // Should panic
    }

    #[test]
    fn test_hadamard_rectangular_matrix() {
        // Test with non-square matrix (3x2)
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let b = Tensor::new(vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0], 3, 2);
        
        let result = a.hadamard(&b);
        let expected = Tensor::new(vec![2.0, 2.0, 12.0, 12.0, 30.0, 30.0], 3, 2);
        
        assert!(tensors_equal(&result, &expected, 1e-10));
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 2);
    }

    #[test]
    fn test_hadamard_compare_with_sequential() {
        // Compare parallel implementation with sequential for verification
        let rows = 20;
        let cols = 15;
        let total = rows * cols;
        
        let a_data: Vec<f32> = (0..total).map(|x| (x as f32) * 0.7 + 1.5).collect();
        let b_data: Vec<f32> = (0..total).map(|x| (x as f32) * 1.3 - 0.5).collect();
        
        let a = Tensor::new(a_data.clone(), rows, cols);
        let b = Tensor::new(b_data.clone(), rows, cols);
        
        // Parallel implementation
        let parallel_result = a.hadamard(&b);
        
        // Sequential implementation for comparison
        let mut sequential_data = Vec::with_capacity(total as usize);
        for i in 0..total {
            sequential_data.push(a_data[i as usize] * b_data[i as usize]);
        }
        let sequential_result = Tensor::new(sequential_data, rows, cols);
        
        assert!(tensors_equal(&parallel_result, &sequential_result, 1e-12));
    }
}