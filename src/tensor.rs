use std::time::{Instant};
use std::{thread, vec};

use rand_pcg::Pcg64;
use rand::distributions::{Distribution, Uniform};

#[derive(Clone, Copy)]
struct RawPointerWrapper {
    raw: *mut f64,
}

unsafe impl Send for RawPointerWrapper {}

#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub rows: u32,
    pub cols: u32
}

impl Tensor {

    pub fn new(data: Vec<f64>, rows: u32, cols: u32) -> Tensor {
        Tensor {
            data: data,
            rows: rows,
            cols: cols
        }
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
                print!("{:.2} ", self.data[(r * self.cols + c) as usize]);
            }
            println!();
        }
    }

    pub fn dims(&self) -> (u32, u32) {
        // println!("({}, {})", self.rows, self.cols)
        (self.rows, self.cols)
    }

    #[allow(dead_code)]
    pub fn modify_vector_chunk(index: usize, val: f64, vec_ptr: RawPointerWrapper) {
        unsafe {
            let ptr = vec_ptr.raw.add(index);
            *ptr = val;
        }
    }

    #[allow(dead_code)]
    pub fn mul_seq(&self, matrix: &Tensor) -> Tensor {
        self.mul_par(matrix, 1)
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

        Tensor::new(result, r1 as u32, c2 as u32)
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
        
        Tensor::new(result, r1 as u32, c2 as u32)
    }

}