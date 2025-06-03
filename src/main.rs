use std::time::{Duration, Instant};
use std::{thread, vec};
use std::sync::{Arc, Mutex};

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

    pub fn mul_par(&self, matrix: &Tensor, nb_threads: usize) -> Tensor {
        let c1 = self.cols as usize;
        let _r1 = self.rows as usize;
        let _c2 = matrix.cols as usize;
        let r2 = matrix.rows as usize;

        let result = Arc::new(Mutex::new(vec![0.0; r2]));
        let nb_threads = nb_threads;
        let chunk_size = r2 / nb_threads;

        let mut handles = vec![];

        for t in 0..nb_threads {
            let start = t * chunk_size;
            let end = if t == nb_threads - 1 { r2 } else { start + chunk_size };

            let a = self.data.clone();
            let b = matrix.data.clone();
            let result = Arc::clone(&result);

            let handle = thread::spawn(move || {
                for i in start..end {
                    // let index = i * c1;
                    let mut sum = 0.0;
                    for j in 0..c1 {
                        sum += a[i * c1 + j] * b[j];
                    }
                    result.lock().unwrap()[i] = sum;
                }
            });
            handles.push(handle);
        }

        // println!("Waiting ...");
        for handle in handles {
            handle.join().unwrap();
        }

        let final_result = Arc::try_unwrap(result).unwrap().into_inner().unwrap();
        Tensor::new(final_result, r2 as u32, 1)
    }

}

fn main() {

    let size = 1000;

    let mut a = vec![];
    for i in 0..size {
        for j in 0..size {
            a.push((i + j) as f64);
        }
    }

    let b: Vec<f64> = (1..=size).map(|x| x as f64).collect();

    let mat1: Tensor = Tensor::new(a, size, size);
    let mat2: Tensor = Tensor::new(b, size, 1);

    let mut total_duration = Duration::new(0, 0);

    for _ in 0..10 {
        let start = Instant::now();
        let _res = mat1.mul_par(&mat2, 1);
        let duration = start.elapsed();
        total_duration += duration;
    }

    let average_duration = total_duration / 10;
    println!("Average time for {size}x{size} * {size}x1 over 10 runs: {:.6?}", average_duration);

}