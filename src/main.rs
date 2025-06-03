use std::time::{Instant};
use std::{thread, vec};

#[derive(Clone, Copy)]
struct RawPointerWrapper {
    raw: *mut f64,
}

unsafe impl Send for RawPointerWrapper {}

fn modify_vector_chunk(index: usize, val: f64, vec_ptr: RawPointerWrapper) {
    unsafe {
        let ptr = vec_ptr.raw.add(index);
        *ptr = val;
    }
}

#[derive(Debug)]
struct Tensor {
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

    pub fn print(&self) {

        for r in 0..self.rows {
            for c in 0.. self.cols {
                print!("{}", self.data[(r * self.cols + c) as usize]);
            }
            println!("");
        }

    }

    pub fn mul_par(&self, matrix: &Tensor, nb_threads: usize) -> Tensor {

        let c1 = self.cols as usize;
        let r1 = self.rows as usize;
        let _c2 = matrix.cols as usize;
        let _r2 = matrix.rows as usize;


        let mut result = vec![0.0; r1];
        let nb_threads = nb_threads;
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
                    let mut sum = 0.0;
                    for j in 0..c1 {
                        sum += a[i * c1 + j] * b[j];
                    }
                    modify_vector_chunk(i, sum, raw_pointer);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        Tensor::new(result, r1 as u32, 1)

    }

    #[cfg(target_arch = "x86_64")]
    pub fn mul_par_simd(&self, matrix: &Tensor, nb_threads: usize) -> Tensor {
        use std::{arch::x86_64::*, sync::Arc};
        
        let c1 = self.cols as usize;
        let r1 = self.rows as usize;
        
        let mut result = vec![0.0; r1];
        let a = Arc::new(&self.data);
        let b = Arc::new(&matrix.data);
        
        let chunk_size = (r1 + nb_threads - 1) / nb_threads;
        
        thread::scope(|s| {
            let chunks: Vec<_> = result
                .chunks_mut(chunk_size)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let a = Arc::clone(&a);
                    let b = Arc::clone(&b);
                    
                    s.spawn(move || {
                        let start_row = chunk_idx * chunk_size;
                        
                        for (local_i, result_elem) in chunk.iter_mut().enumerate() {
                            let i = start_row + local_i;
                            if i >= r1 { break; }
                            
                            let row_start = i * c1;
                            let mut sum = 0.0;
                            
                            unsafe {
                                let mut sum_vec = _mm256_setzero_pd();
                                let mut j = 0;
                                
                                // Process 4 f64s at a time with AVX2
                                while j + 4 <= c1 {
                                    let a_vec = _mm256_loadu_pd(a.as_ptr().add(row_start + j));
                                    let b_vec = _mm256_loadu_pd(b.as_ptr().add(j));
                                    sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
                                    j += 4;
                                }
                                
                                // Extract sum from vector
                                let mut temp = [0.0; 4];
                                _mm256_storeu_pd(temp.as_mut_ptr(), sum_vec);
                                sum = temp.iter().sum();
                                
                                // Handle remaining elements
                                while j < c1 {
                                    sum += a[row_start + j] * b[j];
                                    j += 1;
                                }
                            }
                            
                            *result_elem = sum;
                        }
                    })
                })
                .collect();
        });
        
        Tensor::new(result, r1 as u32, 1)
    }

}

fn main() {

    let size = 5;

    let mut a = vec![];
    for i in 0..size {
        for j in 0..size {
            a.push((i + j) as f64);
        }
    }

    let b: Vec<f64> = (1..=size).map(|x| x as f64).collect();

    let mat1: Tensor = Tensor::new(a, size, size);
    let mat2: Tensor = Tensor::new(b, size, 1);

    let start = Instant::now();
    let _res = mat1.mul_par_simd(&mat2, 16);
    let duration = start.elapsed();

    println!("Average time for {size}x{size} * {size}x1 over 10 runs: {:.6?}", duration);

    _res.print();

}