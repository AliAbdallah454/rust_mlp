use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps};
use std::sync::Arc;
use std::{thread, vec};
use std::ops::{Add, Sub};
use rand_pcg::Pcg64;
use rand::distributions::{Distribution, Uniform};

#[derive(Clone, Copy)]
struct RawPointerWrapper {
    raw: *mut f32,
}

unsafe impl Send for RawPointerWrapper {}

unsafe impl Sync for RawPointerWrapper {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential,
    Parallel(usize),      // number of threads
    SIMD,
    ParallelSIMD(usize),  // number of threads
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.rows, rhs.rows, "Tensor add: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Tensor add: col mismatch");
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a + b).collect();
        Tensor::new(data, self.rows, self.cols)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.rows, rhs.rows, "Tensor sub: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Tensor sub: col mismatch");
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect();
        Tensor::new(data, self.rows, self.cols)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.rows, rhs.rows, "Tensor sub: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Tensor sub: col mismatch");
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect();
        Tensor::new(data, self.rows, self.cols)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let epsilon = 1e-2;
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        self.data.iter()
            .zip(&other.data)
            .all(|(a, b)| (a - b).abs() < epsilon)
    }
}

impl Tensor {

    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
        Tensor {
            data: data,
            rows: rows,
            cols: cols
        }
    }

    pub fn scalar(scalar: f32) -> Tensor {
        Tensor { data: vec![scalar], rows: 1, cols: 1 }
    }

    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = Pcg64::seed_from_u64(seed);
        // can be changed to be from -1.0 to 1.0 later
        let uniform = Uniform::new(0.0, 1.0);
        let data = (0..rows * cols)
            .map(|_| uniform.sample(&mut rng))
            .collect::<Vec<f32>>();

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

    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn ones(rows: usize, cols: usize) -> Tensor {
        let data = vec![1.0; (rows * cols) as usize];
        Tensor::new(data, rows, cols)
    }

    pub fn zeros(rows: usize, cols: usize) -> Tensor {
        let data = vec![0.0; (rows * cols) as usize];
        Tensor::new(data, rows, cols)
    }

    pub fn transpose(&self) -> Tensor {
        let mut data = vec![0.0; (self.rows * self.cols) as usize];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[(j * self.rows + i) as usize] = self.data[(i * self.cols + j) as usize];
            }
        }
        Tensor::new(data, self.cols, self.rows)
    }

    pub fn scale(&self, scalar: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::new(data, self.rows, self.cols)
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn square(&self) -> Tensor {
        let data = self.data.iter().map(|x| x * x).collect();
        Tensor::new(data, self.rows, self.cols)
    }

    #[allow(dead_code)]
    fn modify_vector_chunk(index: usize, val: f32, vec_ptr: RawPointerWrapper) {
        unsafe {
            let ptr = vec_ptr.raw.add(index);
            *ptr = val;
        }
    }

    pub fn mul_vec(&self, vector: &Tensor) -> Tensor {
        assert_eq!(vector.cols, 1, "Expected vector to be rx1 instead of {}x{}", vector.rows, vector.cols);
        let mut res = vec![0.0 as f32; self.rows as usize];

        for i in 0..self.rows {
            unsafe {
                let mut total = 0.0 as f32;
                let mut elem = _mm256_setzero_ps();
                
                let complete_chunks = self.cols / 8;
                for j in 0..complete_chunks {
                    let offset = j * 8;
                    let a_vec = _mm256_loadu_ps(self.data.as_ptr().add(i*self.cols + offset));
                    let b_vec = _mm256_loadu_ps(vector.data.as_ptr().add(offset));
                    let prod = _mm256_mul_ps(a_vec, b_vec);                   
                    elem = _mm256_add_ps(prod, elem);
                }

                let remaining = self.cols % 8;
                if remaining > 0 {
                    let offset = complete_chunks * 8;
                    for j in 0..remaining {
                        total += self.data[i*self.cols + offset + j] * vector.data[offset + j];
                    }
                }

                let mut values = vec![0.0 as f32; 8];
                _mm256_storeu_ps(values.as_mut_ptr(), elem);
                total += values[0] + values[1] + values[2] + values[3] + 
                        values[4] + values[5] + values[6] + values[7];

                *res.as_mut_ptr().add(i) = total;
            }
        }
        Tensor::new(res, self.rows, 1)
    }

    pub fn mul_vec_parallel(&self, vector: &Tensor, nb_threads: usize) -> Tensor {

        let mut res = vec![0.0 as f32; self.rows];
        let raw_ptr = RawPointerWrapper {raw: res.as_mut_ptr()};

        let rows_per_thread = self.rows / nb_threads;

        let self_data: Arc<Vec<f32>> = Arc::from(self.data.clone());
        let vec_data: Arc<Vec<f32>> = Arc::from(vector.data.clone());

        let mut handles = vec![];

        for i in 0..nb_threads {
            
            let start = i * rows_per_thread;
            let mut end = start + rows_per_thread;
            if i == nb_threads - 1 {
                end = self.rows;
            }

            let self_data = Arc::clone(&self_data);
            let vec_data = Arc::clone(&vec_data);
            let raw_ptr = raw_ptr;
            let cols = self.cols;

            let handle = thread::spawn(move || {

                for k in start..end {
                    unsafe {
                        let mut total = 0.0 as f32;
                        let mut elem = _mm256_setzero_ps();
                        
                        let complete_chunks = cols / 8;
                        for j in 0..complete_chunks {
                            let offset = j * 8;
                            let a_vec = _mm256_loadu_ps(self_data.as_ptr().add(k * cols + offset));
                            let b_vec = _mm256_loadu_ps(vec_data.as_ptr().add(offset));
                            let prod = _mm256_mul_ps(a_vec, b_vec);                   
                            elem = _mm256_add_ps(prod, elem);
                        }

                        let remaining = cols % 8;
                        if remaining > 0 {
                            let offset = complete_chunks * 8;
                            for j in 0..remaining {
                                total += self_data[k * cols + offset + j] * vec_data[offset + j];
                            }
                        }
        
                        let mut values = vec![0.0 as f32; 8];
                        _mm256_storeu_ps(values.as_mut_ptr(), elem);
                        total += values[0] + values[1] + values[2] + values[3] + 
                                values[4] + values[5] + values[6] + values[7];
        
                        Tensor::modify_vector_chunk(k, total, raw_ptr);
                    }
                }

            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        Tensor::new(res, self.rows, 1)
    }

    pub fn mul_simd(&self, tensor: &Tensor) -> Tensor {
        assert_eq!(self.cols, tensor.rows, "Expected tensor dimensions to match: {}x{} * {}x{}", self.rows, self.cols, tensor.rows, tensor.cols);
        let mut res = vec![0.0 as f32; (self.rows * tensor.cols) as usize];
    
        let transposed = if tensor.cols != 1 {
            &tensor.transpose()
        } else {
            tensor
        };

        for i in 0..self.rows {
            for k in 0..tensor.cols {
                unsafe {
                    let mut total = 0.0 as f32;
                    let mut elem = _mm256_setzero_ps();
                    
                    let complete_chunks = self.cols / 8;
                    for j in 0..complete_chunks {
                        let offset = j * 8;

                        let a_vec = _mm256_loadu_ps(self.data.as_ptr().add(i * self.cols + offset));
                        let b_vec = _mm256_loadu_ps(transposed.data.as_ptr().add(k * transposed.cols + offset));

                        let prod = _mm256_mul_ps(a_vec, b_vec);                   
                        elem = _mm256_add_ps(prod, elem);
                    }
    
                    let remaining = self.cols % 8;
                    if remaining > 0 {
                        let offset = complete_chunks * 8;
                        for j in 0..remaining {
                            total += self.data[i * self.cols + offset + j] * transposed.data[k * transposed.cols + offset + j];
                        }
                    }
    
                    let mut values = [0.0f32; 8];
                    _mm256_storeu_ps(values.as_mut_ptr(), elem);
                    total += values[0] + values[1] + values[2] + values[3] + 
                            values[4] + values[5] + values[6] + values[7];
    
                    *res.as_mut_ptr().add(i * tensor.cols + k) = total;
                }
            }
        }
        Tensor::new(res, self.rows, tensor.cols)
    }

    pub fn mul_simd_parallel(&self, tensor: &Tensor, nb_threads: usize) -> Tensor {

        let mut flag = true;

        let tensor = if tensor.cols != 1 {
            flag = false;
            &tensor.transpose()
        } else {
            &tensor.clone()
        };

        let c1 = self.cols;
        let r1 = self.rows;

        let mut r2 = 0;
        let mut c2 = 0;

        if flag {
            r2 = tensor.rows;
            c2 = tensor.cols;
        } else {
            r2 = tensor.cols;
            c2 = tensor.rows;
        }

        let mut res = vec![0.0 as f32; r1 * c2];
        let raw_ptr = RawPointerWrapper {raw: res.as_mut_ptr()};

        let rows_per_thread = r1 / nb_threads;
        
        let self_data: Arc<Vec<f32>> = Arc::from(self.data.clone());
        let tensor_data: Arc<Vec<f32>> = Arc::from(tensor.data.clone());

        let mut handles = vec![];

        for i in 0..nb_threads {
            
            let start = i * rows_per_thread;
            let mut end = start + rows_per_thread;
            if i == nb_threads - 1 {
                end = self.rows;
            }

            let self_data = Arc::clone(&self_data);
            let vec_data = Arc::clone(&tensor_data);
            let raw_ptr = raw_ptr;

            let handle = thread::spawn(move || {

                for i in start..end {
                    for k in 0..c2 {
                        unsafe {
                            let mut total = 0.0 as f32;
                            let mut elem = _mm256_setzero_ps();
                            
                            let complete_chunks = c1 / 8;
                            for j in 0..complete_chunks {
                                let offset = j * 8;
        
                                let a_vec = _mm256_loadu_ps(self_data.as_ptr().add(i * c1 + offset));
                                let b_vec = _mm256_loadu_ps(vec_data.as_ptr().add(k * r2 + offset));
                                // let b_vec = _mm256_loadu_ps(vec_data.as_ptr().add(k * c2 + offset));
        
                                let prod = _mm256_mul_ps(a_vec, b_vec);                   
                                elem = _mm256_add_ps(prod, elem);
                            }
            
                            let remaining = c1 % 8;
                            if remaining > 0 {
                                let offset = complete_chunks * 8;
                                for j in 0..remaining {
                                    // total += self_data[i * c1 + offset + j] * vec_data[k * c2 + offset + j];
                                    total += self_data[i * c1 + offset + j] * vec_data[k * r2 + offset + j];
                                }
                            }
            
                            let mut values = [0.0f32; 8];
                            _mm256_storeu_ps(values.as_mut_ptr(), elem);
                            total += values[0] + values[1] + values[2] + values[3] + 
                                    values[4] + values[5] + values[6] + values[7];
            
                            Tensor::modify_vector_chunk(i * c2 + k, total, raw_ptr);
                        }
                    }
                }

            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        Tensor::new(res, r1, c2)

    }

    #[allow(dead_code)]
    pub fn mul_seq(&self, matrix: &Tensor) -> Tensor {

        let c1 = self.cols;
        let r1 = self.rows;
        let c2 = matrix.cols;
        let r2 = matrix.rows;

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
        Tensor::new(result, r1, c2)
    }

    #[allow(dead_code)]
    pub fn mul_par(&self, matrix: &Tensor, nb_threads: usize) -> Tensor {
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

        Tensor::new(result, r1, c2)
    }

    pub fn mul(&self, matrix: &Tensor, execution_mode: ExecutionMode) -> Tensor {
        match execution_mode {
            ExecutionMode::Sequential => self.mul_seq(matrix),
            ExecutionMode::Parallel(threads) => self.mul_par(matrix, threads),
            ExecutionMode::SIMD => self.mul_simd(matrix),
            ExecutionMode::ParallelSIMD(threads) => self.mul_simd_parallel(matrix, threads)
        }
    }

    pub fn argmax(&self) -> usize {
        assert_eq!(self.cols, 1, "argmax only works on column vectors (rx1 tensors)");
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

    // Element wise multiplication (i saw that the element wise mulitplication is called hadamard multiplication hence the name :) )
    pub fn hadamard(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows, "Tensor hadamard: row mismatch");
        assert_eq!(self.cols, other.cols, "Tensor hadamard: col mismatch");
        
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Tensor::new(data, self.rows, self.cols)
    }

    // Sequential implementation
    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        Tensor::new(data, self.rows, self.cols)
    }

    pub fn relu_derivative(&self) -> Tensor {
        let data = self.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        Tensor::new(data, self.rows, self.cols)
    }

    pub fn sigmoid(&self) -> Tensor {
        let data = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::new(data, self.rows, self.cols)
    }

    /// Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    pub fn sigmoid_derivative(&self) -> Tensor {
        let sigmoid_vals = self.sigmoid();
        let ones = Tensor::ones(self.rows, self.cols);
        let one_minus_sigmoid = &ones - &sigmoid_vals;
        sigmoid_vals.hadamard(&one_minus_sigmoid)
    }

    pub fn tanh(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor::new(data, self.rows, self.cols)
    }

    /// Tanh derivative: 1 - tanh(x)^2
    pub fn tanh_derivative(&self) -> Tensor {
        let tanh_vals = self.tanh();
        let ones = Tensor::ones(self.rows, self.cols);
        let tanh_squared = tanh_vals.hadamard(&tanh_vals);
        &ones - &tanh_squared
    }

    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.cols, 1, "Softmax only implemented for column vectors (r x 1)");

        let exp_vals: Vec<f32> = self.data.iter().map(|&x| x.exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let data: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();
        Tensor::new(data, self.rows, self.cols)
    }

    // to get the derivative of the softmax we need to calculate its Jacobian Matrix
    pub fn softmax_derivative(&self) -> Tensor {
        assert_eq!(self.cols, 1, "Softmax derivative only implemented for column vectors (r x 1)");
    
        let softmax = self.softmax();
        let len = softmax.data.len();
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
    
        Tensor::new(jacobian, len as usize, len as usize) // Jacobian is (r x r)
    }

    pub fn mse_loss(&self, target: &Tensor) -> f32 {
        let diff = self - target;
        let squared = diff.square();
        squared.sum() / (self.rows * self.cols) as f32
    }

    pub fn mse_loss_derivative(&self, target: &Tensor) -> Tensor {
        let diff = self - target;
        diff.scale(2.0 / (self.rows * self.cols) as f32)
    }

    // sum of -y*log(y_hat)
    pub fn categorical_cross_entropy(&self, targets: &Tensor) -> f32 {
        assert_eq!(self.rows, targets.rows, "Batch size mismatch");
        assert_eq!(self.cols, targets.cols, "Class count mismatch");
    
        let epsilon = 1e-15; // To prevent log(0)
        let mut total_loss = 0.0;
    
        for i in 0..(self.rows * self.cols) as usize {
            let y_true = targets.data[i];
            let y_pred = self.data[i].max(epsilon).min(1.0 - epsilon); // clip
            total_loss -= y_true * y_pred.ln();
        }
    
        total_loss / self.rows as f32 // Can be changed ...
    }
    
    // softmax - y
    pub fn categorical_cross_entropy_derivative(&self, targets: &Tensor) -> Tensor {
        assert_eq!(self.rows, targets.rows, "Batch size mismatch");
        assert_eq!(self.cols, targets.cols, "Class count mismatch");
        assert_eq!(self.cols, 1, "Categorical cross entropy derivative only implemented for column vectors");

        // self: softmax output, targets: one-hot labels
        let diff = self - targets;
        diff.scale(1.0 / self.rows as f32) // Can be changed ...
    }

}