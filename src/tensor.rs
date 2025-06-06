use std::{thread, vec};
use std::ops::{Add, Sub};

use rand_pcg::Pcg64;
use rand::distributions::{Distribution, Uniform};

use rayon::prelude::*;

#[derive(Clone, Copy)]
struct RawPointerWrapper {
    raw: *mut f64,
}

unsafe impl Send for RawPointerWrapper {}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub rows: u32,
    pub cols: u32,
    pub concurrent: bool
}

// Overload the + operator for Tensor + Tensor
impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.rows, rhs.rows, "Tensor add: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Tensor add: col mismatch");
        
        let data = if self.concurrent {
            self.data.par_iter().zip(rhs.data.par_iter()).map(|(a, b)| a + b).collect()
        } else {
            self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a + b).collect()
        };
        
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }
}

// Overload the - operator for Tensor - Tensor
impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.rows, rhs.rows, "Tensor sub: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Tensor sub: col mismatch");
        
        let data = if self.concurrent {
            self.data.par_iter().zip(rhs.data.par_iter()).map(|(a, b)| a - b).collect()
        } else {
            self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect()
        };
        
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.rows, rhs.rows, "Tensor sub: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Tensor sub: col mismatch");
        
        let concurrent = self.concurrent;
        let data = if concurrent {
            self.data.par_iter().zip(rhs.data.par_iter()).map(|(a, b)| a - b).collect()
        } else {
            self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect()
        };
        
        Tensor::new_with_concurrent(data, self.rows, self.cols, concurrent)
    }
}

impl Tensor {

    pub fn new(data: Vec<f64>, rows: u32, cols: u32) -> Tensor {
        Tensor {
            data: data,
            rows: rows,
            cols: cols,
            concurrent: true
        }
    }

    pub fn new_with_concurrent(data: Vec<f64>, rows: u32, cols: u32, concurrent: bool) -> Tensor {
        Tensor {
            data: data,
            rows: rows,
            cols: cols,
            concurrent: concurrent
        }
    }

    pub fn set_concurrent(&mut self, val: bool) {
        self.concurrent = val;
    }

    pub fn scalar(scalar: f64) -> Tensor {
        Tensor::new(vec![scalar], 1, 1)
    }

    pub fn random(rows: u32, cols: u32, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = Pcg64::seed_from_u64(seed);
        let uniform = Uniform::new(0.0, 1.0);
        let data = (0..rows * cols)
            .map(|_| uniform.sample(&mut rng))
            .collect::<Vec<f64>>();

        Tensor::new(data, rows, cols)
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                print!("{:.5} ", self.data[(r * self.cols + c) as usize]);
            }
            println!();
        }
    }

    pub fn dims(&self) -> (u32, u32) {
        (self.rows, self.cols)
    }

    #[allow(dead_code)]
    fn modify_vector_chunk(index: usize, val: f64, vec_ptr: RawPointerWrapper) {
        unsafe {
            let ptr = vec_ptr.raw.add(index);
            *ptr = val;
        }
    }

    pub fn mul_seq(&self, matrix: &Tensor) -> Tensor {
        let c1 = self.cols as usize;
        let r1 = self.rows as usize;
        let c2 = matrix.cols as usize;
        let r2 = matrix.rows as usize;
    
        assert_eq!(c1, r2, "Matrix dimensions don't match: {}x{} * {}x{}", r1, c1, r2, c2);
    
        let mut result = vec![0.0; r1 * c2];
    
        for i in 0..r1 {
            for j in 0..c2 {
                let mut sum = 0.0;
                for k in 0..c1 {
                    sum += self.data[i * c1 + k] * matrix.data[k * c2 + j];
                }
                result[i * c2 + j] = sum;
            }
        }
    
        Tensor::new_with_concurrent(result, r1 as u32, c2 as u32, self.concurrent)
    }

    pub fn mul_par(&self, matrix: &Tensor) -> Tensor {
        let c1 = self.cols as usize;
        let r1 = self.rows as usize;
        let c2 = matrix.cols as usize;
        let r2 = matrix.rows as usize;
    
        assert_eq!(c1, r2, "Matrix dimensions don't match: {}x{} * {}x{}", r1, c1, r2, c2);
    
        let result: Vec<f64> = (0..r1)
            .into_par_iter()
            .flat_map(|i| {
                (0..c2).into_par_iter().map(move |j| {
                    (0..c1)
                        .map(|k| self.data[i * c1 + k] * matrix.data[k * c2 + j])
                        .sum()
                })
            })
            .collect();
    
        Tensor::new_with_concurrent(result, r1 as u32, c2 as u32, self.concurrent)
    }

    pub fn mul(&self, matrix: &Tensor) -> Tensor {
        if self.concurrent {
            self.mul_par(matrix)
        } else {
            self.mul_seq(matrix)
        }
    }

    #[allow(dead_code)]
    pub fn mul_par_old(&self, matrix: &Tensor, nb_threads: usize) -> Tensor {
        let c1 = self.cols as usize;
        let r1 = self.rows as usize;
        let c2 = matrix.cols as usize;
        let r2 = matrix.rows as usize;

        assert_eq!(c1, r2, "Matrix dimensions don't match: {}x{} * {}x{}", r1, c1, r2, c2);

        let mut result = vec![0.0; r1 * c2];
        let chunk_size = r1 / nb_threads;

        let mut handles = vec![];

        for t in 0..nb_threads {
            let start = t * chunk_size;
            let end = if t == nb_threads - 1 { r1 } else { start + chunk_size };

            let raw_pointer = RawPointerWrapper {
                raw: result.as_mut_ptr()
            };

            let a = self.data.clone();
            let b = matrix.data.clone();

            let handle = thread::spawn(move || {
                for i in start..end {
                    for j in 0..c2 {
                        let mut sum = 0.0;
                        for k in 0..c1 {
                            sum += a[i * c1 + k] * b[k * c2 + j];
                        }
                        Tensor::modify_vector_chunk(i * c2 + j, sum, raw_pointer);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        Tensor::new_with_concurrent(result, r1 as u32, c2 as u32, self.concurrent)
    }

    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    pub fn mul_par_simd(&self, matrix: &Tensor, nb_threads: usize) -> Tensor {
        use std::{arch::x86_64::*, sync::Arc};
        
        let c1 = self.cols as usize;
        let r1 = self.rows as usize;
        let c2 = matrix.cols as usize;
        let r2 = matrix.rows as usize;

        assert_eq!(c1, r2, "Matrix dimensions don't match: {}x{} * {}x{}", r1, c1, r2, c2);
        
        let mut result = vec![0.0; r1 * c2];
        let a = Arc::new(&self.data);
        let b = Arc::new(&matrix.data);
        
        let chunk_size = (r1 + nb_threads - 1) / nb_threads;
        
        thread::scope(|s| {
            let chunks: Vec<_> = result
                .chunks_mut(chunk_size * c2)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let a = Arc::clone(&a);
                    let b = Arc::clone(&b);
                    
                    s.spawn(move || {
                        let start_row = chunk_idx * chunk_size;
                        let rows_in_chunk = chunk.len() / c2;
                        
                        for local_i in 0..rows_in_chunk {
                            let i = start_row + local_i;
                            if i >= r1 { break; }
                            
                            for j in 0..c2 {
                                let mut sum = 0.0;
                                
                                unsafe {
                                    let mut sum_vec = _mm256_setzero_pd();
                                    let mut k = 0;
                                    
                                    // Process 4 f64s at a time with AVX2
                                    while k + 4 <= c1 {
                                        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i * c1 + k));
                                        let b_vec = _mm256_set_pd(
                                            b[(k + 3) * c2 + j],
                                            b[(k + 2) * c2 + j], 
                                            b[(k + 1) * c2 + j],
                                            b[k * c2 + j]
                                        );
                                        sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
                                        k += 4;
                                    }
                                    
                                    // Extract sum from vector
                                    let mut temp = [0.0; 4];
                                    _mm256_storeu_pd(temp.as_mut_ptr(), sum_vec);
                                    sum = temp.iter().sum();
                                    
                                    // Handle remaining elements
                                    while k < c1 {
                                        sum += a[i * c1 + k] * b[k * c2 + j];
                                        k += 1;
                                    }
                                }
                                
                                chunk[local_i * c2 + j] = sum;
                            }
                        }
                    })
                })
                .collect();
        });
        
        Tensor::new_with_concurrent(result, r1 as u32, c2 as u32, self.concurrent)
    }

    pub fn square(&self) -> Tensor {
        let data = if self.concurrent {
            self.data.par_iter().map(|x| x * x).collect()
        } else {
            self.data.iter().map(|x| x * x).collect()
        };
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }

    pub fn argmax(&self) -> usize {
        assert_eq!(self.cols, 1, "argmax only works on column vectors (rx1 tensors)");
        
        if self.concurrent {
            self.data.par_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            let mut max_idx = 0;
            let mut max_val = self.data[0];
            for (i, &val) in self.data.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            max_idx
        }
    }

    // Element wise multiplication (i saw that the element wise mulitplication is called hadamard multiplication hence the name :) )
    pub fn hadamard(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows, "Tensor hadamard: row mismatch");
        assert_eq!(self.cols, other.cols, "Tensor hadamard: col mismatch");
        
        let data = if self.concurrent {
            self.data.par_iter().zip(other.data.par_iter()).map(|(a, b)| a * b).collect()
        } else {
            self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect()
        };
        
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }

    pub fn relu(&self) -> Tensor {
        let data = if self.concurrent {
            self.data.par_iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
        } else {
            self.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
        };
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }

    /// ReLU derivative (1 if x > 0, 0 otherwise)
    pub fn relu_derivative(&self) -> Tensor {
        let data = if self.concurrent {
            self.data.par_iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect()
        } else {
            self.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect()
        };
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }

    pub fn sigmoid(&self) -> Tensor {
        let data = if self.concurrent {
            self.data.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
        } else {
            self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
        };
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }

    /// Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    pub fn sigmoid_derivative(&self) -> Tensor {
        let sigmoid_vals = self.sigmoid();
        let ones = Tensor::ones_with_concurrent(self.rows, self.cols, self.concurrent);
        let one_minus_sigmoid = &ones - &sigmoid_vals;
        sigmoid_vals.hadamard(&one_minus_sigmoid)
    }

    pub fn tanh(&self) -> Tensor {
        let data = if self.concurrent {
            self.data.par_iter().map(|&x| x.tanh()).collect()
        } else {
            self.data.iter().map(|&x| x.tanh()).collect()
        };
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }

    /// Tanh derivative: 1 - tanh(x)^2
    pub fn tanh_derivative(&self) -> Tensor {
        let tanh_vals = self.tanh();
        let ones = Tensor::ones_with_concurrent(self.rows, self.cols, self.concurrent);
        let tanh_squared = tanh_vals.hadamard(&tanh_vals);
        &ones - &tanh_squared
    }

    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.cols, 1, "Softmax only implemented for column vectors (r x 1)");

        let exp_vals: Vec<f64> = if self.concurrent {
            self.data.par_iter().map(|&x| x.exp()).collect()
        } else {
            self.data.iter().map(|&x| x.exp()).collect()
        };
        
        let sum: f64 = if self.concurrent {
            exp_vals.par_iter().sum()
        } else {
            exp_vals.iter().sum()
        };
        
        let data: Vec<f64> = if self.concurrent {
            exp_vals.par_iter().map(|&x| x / sum).collect()
        } else {
            exp_vals.iter().map(|&x| x / sum).collect()
        };
        
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }

    // to get the derivative of the softmax we need to calculate its Jacobian Matrix
    pub fn softmax_derivative(&self) -> Tensor {
        assert_eq!(self.cols, 1, "Softmax derivative only implemented for column vectors (r x 1)");
    
        let softmax = self.softmax();
        let len = softmax.data.len();
        
        let jacobian = if self.concurrent {
            (0..len).into_par_iter().flat_map(|i| {
                let x = softmax.clone();
                (0..len).into_par_iter().map(move |j| {
                    let s_i = x.data[i];
                    let s_j = x.data[j];
                    if i == j {
                        s_i * (1.0 - s_i)
                    } else {
                        -s_i * s_j
                    }
                })
            }).collect()
        } else {
            let mut jacobian = vec![0.0; len * len];
            for i in 0..len {
                for j in 0..len {
                    let s_i = softmax.data[i];
                    let s_j = softmax.data[j];
                    let val = if i == j {
                        s_i * (1.0 - s_i)
                    } else {
                        -s_i * s_j
                    };
                    jacobian[i * len + j] = val;
                }
            }
            jacobian
        };
    
        Tensor::new_with_concurrent(jacobian, len as u32, len as u32, self.concurrent)
    }

    pub fn ones(rows: u32, cols: u32) -> Tensor {
        let data = vec![1.0; (rows * cols) as usize];
        Tensor::new(data, rows, cols)
    }

    pub fn ones_with_concurrent(rows: u32, cols: u32, concurrent: bool) -> Tensor {
        let data = vec![1.0; (rows * cols) as usize];
        Tensor::new_with_concurrent(data, rows, cols, concurrent)
    }

    pub fn zeros(rows: u32, cols: u32) -> Tensor {
        let data = vec![0.0; (rows * cols) as usize];
        Tensor::new(data, rows, cols)
    }

    pub fn zeros_with_concurrent(rows: u32, cols: u32, concurrent: bool) -> Tensor {
        let data = vec![0.0; (rows * cols) as usize];
        Tensor::new_with_concurrent(data, rows, cols, concurrent)
    }

    pub fn transpose(&self) -> Tensor {
        let size = (self.rows * self.cols) as usize;
        
        let data = if self.concurrent && size > 1000 { // Use parallel for larger matrices
            (0..self.rows).into_par_iter().flat_map(|i| {
                (0..self.cols).into_par_iter().map(move |j| {
                    self.data[(i * self.cols + j) as usize]
                })
            }).collect::<Vec<_>>()
            .into_iter()
            .enumerate()
            .fold(vec![0.0; size], |mut acc, (idx, val)| {
                let orig_i = idx / self.cols as usize;
                let orig_j = idx % self.cols as usize;
                acc[orig_j * self.rows as usize + orig_i] = val;
                acc
            })
        } else {
            let mut data = vec![0.0; size];
            for i in 0..self.rows {
                for j in 0..self.cols {
                    data[(j * self.rows + i) as usize] = self.data[(i * self.cols + j) as usize];
                }
            }
            data
        };
        
        Tensor::new_with_concurrent(data, self.cols, self.rows, self.concurrent)
    }

    pub fn scale(&self, scalar: f64) -> Tensor {
        let data = if self.concurrent {
            self.data.par_iter().map(|&x| x * scalar).collect()
        } else {
            self.data.iter().map(|&x| x * scalar).collect()
        };
        Tensor::new_with_concurrent(data, self.rows, self.cols, self.concurrent)
    }

    pub fn sum(&self) -> f64 {
        if self.concurrent {
            self.data.par_iter().sum()
        } else {
            self.data.iter().sum()
        }
    }

    pub fn mse_loss(&self, target: &Tensor) -> f64 {
        let diff = self - target;
        let squared = diff.square();
        squared.sum() / (self.rows * self.cols) as f64
    }

    pub fn mse_loss_derivative(&self, target: &Tensor) -> Tensor {
        let diff = self - target;
        diff.scale(2.0 / (self.rows * self.cols) as f64)
    }

    // sum of -y*log(y_hat)
    pub fn categorical_cross_entropy(&self, targets: &Tensor) -> f64 {
        assert_eq!(self.rows, targets.rows, "Batch size mismatch");
        assert_eq!(self.cols, targets.cols, "Class count mismatch");
    
        let epsilon = 1e-15; // To prevent log(0)
        
        let total_loss = if self.concurrent {
            (0..(self.rows * self.cols) as usize)
                .into_par_iter()
                .map(|i| {
                    let y_true = targets.data[i];
                    let y_pred = self.data[i].max(epsilon).min(1.0 - epsilon); // clip
                    -y_true * y_pred.ln()
                })
                .sum::<f64>()
        } else {
            let mut total_loss = 0.0;
            for i in 0..(self.rows * self.cols) as usize {
                let y_true = targets.data[i];
                let y_pred = self.data[i].max(epsilon).min(1.0 - epsilon); // clip
                total_loss -= y_true * y_pred.ln();
            }
            total_loss
        };
    
        total_loss / self.rows as f64
    }
    
    // softmax - y
    pub fn categorical_cross_entropy_derivative(&self, targets: &Tensor) -> Tensor {
        assert_eq!(self.rows, targets.rows, "Batch size mismatch");
        assert_eq!(self.cols, targets.cols, "Class count mismatch");
        assert_eq!(self.cols, 1, "Categorical cross entropy derivative only implemented for column vectors");

        // self: softmax output, targets: one-hot labels
        let diff = self - targets;
        diff.scale(1.0 / self.rows as f64)
    }

}