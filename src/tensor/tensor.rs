use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps};
use std::sync::Arc;
use std::{thread, vec};
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
    Parallel,
    SIMD,
    ParallelSIMD,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let expected_size: usize = shape.iter().product();
        assert_eq!(data.len(), expected_size, 
            "Data length {} doesn't match shape {:?} (expected {})", 
            data.len(), shape, expected_size);
        Tensor {
            data,
            shape,
        }
    }

    pub fn scalar(scalar: f32) -> Tensor {
        Tensor { data: vec![scalar], shape: vec![1] }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rows(&self) -> usize {
        if self.shape.len() >= 1 { self.shape[0] } else { 1 }
    }

    pub fn cols(&self) -> usize {
        if self.shape.len() >= 2 { self.shape[1] } else { 1 }
    }

    pub fn is_vector(&self) -> bool {
        self.rank() == 1
    }

    pub fn is_matrix(&self) -> bool {
        self.rank() == 2
    }

    pub fn is_scalar(&self) -> bool {
        self.size() == 1
    }

    pub fn random(shape: Vec<usize>, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = Pcg64::seed_from_u64(seed);
        // can be changed to be from -1.0 to 1.0 later
        let uniform = Uniform::new(0.0, 1.0);
        let size: usize = shape.iter().product();
        let data = (0..size)
            .map(|_| uniform.sample(&mut rng))
            .collect::<Vec<f32>>();

        Tensor::new(data, shape)
    }

    pub fn print(&self) {
        if self.rank() == 2 {
            // Print as matrix
            for r in 0..self.rows() {
                for c in 0..self.cols() {
                    print!("{:.5} ", self.data[(r * self.cols() + c) as usize]);
                }
                println!();
            }
        } else {
            // Print as general tensor with shape
            println!("Tensor shape: {:?}", self.shape);
            println!("Data: {:?}", self.data);
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    pub fn ones(shape: Vec<usize>) -> Tensor {
        let size: usize = shape.iter().product();
        let data = vec![1.0; size];
        Tensor::new(data, shape)
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        Tensor::new(data, shape)
    }

    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.rank(), 2, "Transpose only supported for 2D tensors");
        let rows = self.rows();
        let cols = self.cols();
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
            }
        }
        Tensor::new(data, vec![cols, rows])
    }

    pub fn scale(&self, scalar: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn square(&self) -> Tensor {
        let data = self.data.iter().map(|x| x * x).collect();
        Tensor::new(data, self.shape.clone())
    }

    #[allow(dead_code)]
    fn modify_vector_chunk(index: usize, val: f32, vec_ptr: RawPointerWrapper) {
        unsafe {
            let ptr = vec_ptr.raw.add(index);
            *ptr = val;
        }
    }

    pub fn mul_vec(&self, vector: &Tensor) -> Tensor {
        let rows = self.rows();
        let cols = self.cols();
        let vec_cols = vector.cols();
        assert_eq!(vec_cols, 1, "Expected vector to be rx1 instead of {}x{}", vector.rows(), vec_cols);
        let mut res = vec![0.0 as f32; rows];

        for i in 0..rows {
            unsafe {
                let mut total = 0.0 as f32;
                let mut elem = _mm256_setzero_ps();
                
                let complete_chunks = cols / 8;
                for j in 0..complete_chunks {
                    let offset = j * 8;
                    let a_vec = _mm256_loadu_ps(self.data.as_ptr().add(i*cols + offset));
                    let b_vec = _mm256_loadu_ps(vector.data.as_ptr().add(offset));
                    let prod = _mm256_mul_ps(a_vec, b_vec);                   
                    elem = _mm256_add_ps(prod, elem);
                }

                let remaining = cols % 8;
                if remaining > 0 {
                    let offset = complete_chunks * 8;
                    for j in 0..remaining {
                        total += self.data[i*cols + offset + j] * vector.data[offset + j];
                    }
                }

                let mut values = vec![0.0 as f32; 8];
                _mm256_storeu_ps(values.as_mut_ptr(), elem);
                total += values[0] + values[1] + values[2] + values[3] + 
                        values[4] + values[5] + values[6] + values[7];

                *res.as_mut_ptr().add(i) = total;
            }
        }
        Tensor::new(res, vec![rows, 1])
    }

    pub fn mul_vec_parallel(&self, vector: &Tensor, nb_threads: usize) -> Tensor {
        let rows = self.rows();
        let cols = self.cols();

        let mut res = vec![0.0 as f32; rows];
        let raw_ptr = RawPointerWrapper {raw: res.as_mut_ptr()};

        let rows_per_thread = rows / nb_threads;

        let self_data: Arc<Vec<f32>> = Arc::from(self.data.clone());
        let vec_data: Arc<Vec<f32>> = Arc::from(vector.data.clone());

        let mut handles = vec![];

        for i in 0..nb_threads {
            
            let start = i * rows_per_thread;
            let mut end = start + rows_per_thread;
            if i == nb_threads - 1 {
                end = rows;
            }

            let self_data = Arc::clone(&self_data);
            let vec_data = Arc::clone(&vec_data);
            let raw_ptr = raw_ptr;

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
        Tensor::new(res, vec![rows, 1])
    }

    pub fn mul_simd(&self, tensor: &Tensor) -> Tensor {
        let self_rows = self.rows();
        let self_cols = self.cols();
        let tensor_rows = tensor.rows();
        let tensor_cols = tensor.cols();
        
        assert_eq!(self_cols, tensor_rows, "Expected tensor dimensions to match: {}x{} * {}x{}", self_rows, self_cols, tensor_rows, tensor_cols);
        let mut res = vec![0.0 as f32; self_rows * tensor_cols];
    
        let transposed = if tensor_cols != 1 {
            &tensor.transpose()
        } else {
            tensor
        };

        for i in 0..self_rows {
            for k in 0..tensor_cols {
                unsafe {
                    let mut total = 0.0 as f32;
                    let mut elem = _mm256_setzero_ps();
                    
                    let complete_chunks = self_cols / 8;
                    for j in 0..complete_chunks {
                        let offset = j * 8;

                        let a_vec = _mm256_loadu_ps(self.data.as_ptr().add(i * self_cols + offset));
                        let b_vec = _mm256_loadu_ps(transposed.data.as_ptr().add(k * transposed.cols() + offset));

                        let prod = _mm256_mul_ps(a_vec, b_vec);                   
                        elem = _mm256_add_ps(prod, elem);
                    }
    
                    let remaining = self_cols % 8;
                    if remaining > 0 {
                        let offset = complete_chunks * 8;
                        for j in 0..remaining {
                            total += self.data[i * self_cols + offset + j] * transposed.data[k * transposed.cols() + offset + j];
                        }
                    }
    
                    let mut values = [0.0f32; 8];
                    _mm256_storeu_ps(values.as_mut_ptr(), elem);
                    total += values[0] + values[1] + values[2] + values[3] + 
                            values[4] + values[5] + values[6] + values[7];
    
                    *res.as_mut_ptr().add(i * tensor_cols + k) = total;
                }
            }
        }
        Tensor::new(res, vec![self_rows, tensor_cols])
    }

    pub fn mul_simd_parallel(&self, tensor: &Tensor, nb_threads: usize) -> Tensor {

        let mut flag = true;

        let tensor = if tensor.cols() != 1 {
            flag = false;
            &tensor.transpose()
        } else {
            &tensor.clone()
        };

        let c1 = self.cols();
        let r1 = self.rows();

        let mut r2 = 0;
        let mut c2 = 0;

        if flag {
            r2 = tensor.rows();
            c2 = tensor.cols();
        } else {
            r2 = tensor.cols();
            c2 = tensor.rows();
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
                end = r1;
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
        Tensor::new(res, vec![r1, c2])

    }

    #[allow(dead_code)]
    pub fn mul_seq(&self, matrix: &Tensor) -> Tensor {

        let c1 = self.cols();
        let r1 = self.rows();
        let c2 = matrix.cols();
        let r2 = matrix.rows();

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
        Tensor::new(result, vec![r1, c2])
    }

    #[allow(dead_code)]
    pub fn mul_par(&self, matrix: &Tensor, nb_threads: usize) -> Tensor {
        let c1 = self.cols();
        let r1 = self.rows();
        let c2 = matrix.cols();
        let r2 = matrix.rows();

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

        Tensor::new(result, vec![r1, c2])
    }

    pub fn mul(&self, matrix: &Tensor, execution_mode: ExecutionMode) -> Tensor {
        match execution_mode {
            ExecutionMode::Sequential => self.mul_seq(matrix),
            ExecutionMode::Parallel => self.mul_par(matrix, 6),
            ExecutionMode::SIMD => self.mul_simd(matrix),
            ExecutionMode::ParallelSIMD => self.mul_simd_parallel(matrix, 6)
        }
    }

    pub fn argmax(&self) -> usize {
        assert_eq!(self.cols(), 1, "argmax only works on column vectors (rx1 tensors)");
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
        assert_eq!(self.shape, other.shape, "Tensor hadamard: shape mismatch {:?} vs {:?}", self.shape, other.shape);
        
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Tensor::new(data, self.shape.clone())
    }

    // Sequential implementation
    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn relu_derivative(&self) -> Tensor {
        let data = self.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sigmoid(&self) -> Tensor {
        let data = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    pub fn sigmoid_derivative(&self) -> Tensor {
        let sigmoid_vals = self.sigmoid();
        let ones = Tensor::ones(self.shape.clone());
        let one_minus_sigmoid = &ones - &sigmoid_vals;
        sigmoid_vals.hadamard(&one_minus_sigmoid)
    }

    pub fn tanh(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Tanh derivative: 1 - tanh(x)^2
    pub fn tanh_derivative(&self) -> Tensor {
        let tanh_vals = self.tanh();
        let ones = Tensor::ones(self.shape.clone());
        let tanh_squared = tanh_vals.hadamard(&tanh_vals);
        &ones - &tanh_squared
    }

    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.cols(), 1, "Softmax only implemented for column vectors (r x 1)");

        let exp_vals: Vec<f32> = self.data.iter().map(|&x| x.exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let data: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();
        Tensor::new(data, self.shape.clone())
    }

    // to get the derivative of the softmax we need to calculate its Jacobian Matrix
    pub fn softmax_derivative(&self) -> Tensor {
        assert_eq!(self.cols(), 1, "Softmax derivative only implemented for column vectors (r x 1)");
    
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
    
        Tensor::new(jacobian, vec![len, len]) // Jacobian is (r x r)
    }

    pub fn mse_loss(&self, target: &Tensor) -> f32 {
        let diff = self - target;
        let squared = diff.square();
        squared.sum() / self.size() as f32
    }

    pub fn mse_loss_derivative(&self, target: &Tensor) -> Tensor {
        let diff = self - target;
        diff.scale(2.0 / self.size() as f32)
    }

    // sum of -y*log(y_hat)
    pub fn categorical_cross_entropy(&self, targets: &Tensor) -> f32 {
        assert_eq!(self.shape, targets.shape, "Shape mismatch {:?} vs {:?}", self.shape, targets.shape);
    
        let epsilon = 1e-15; // To prevent log(0)
        let mut total_loss = 0.0;
    
        for i in 0..self.size() {
            let y_true = targets.data[i];
            let y_pred = self.data[i].max(epsilon).min(1.0 - epsilon); // clip
            total_loss -= y_true * y_pred.ln();
        }
    
        total_loss / self.rows() as f32 // Can be changed ...
    }
    
    // softmax - y
    pub fn categorical_cross_entropy_derivative(&self, targets: &Tensor) -> Tensor {
        assert_eq!(self.shape, targets.shape, "Shape mismatch {:?} vs {:?}", self.shape, targets.shape);
        assert_eq!(self.cols(), 1, "Categorical cross entropy derivative only implemented for column vectors");

        // self: softmax output, targets: one-hot labels
        let diff = self - targets;
        diff.scale(1.0 / self.rows() as f32) // Can be changed ...
    }

}