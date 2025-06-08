use cp_proj::tensor::Tensor;
use std::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};
use std::time::Instant;

fn _side_by_size(a: &Tensor, b: &Tensor) {
    assert_eq!(a.data.len(), b.data.len(), "Tensor data lengths must match");

    for i in 0..a.data.len() {
        println!("{} - {}: {}", a.data[i], b.data[i], a.data[i] == b.data[i]);
    }

}

unsafe fn _element_wise_mul(a: &Vec<f32>, b: &Vec<f32>, c: *mut f32) {

    let len = a.len();

    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        unsafe {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(idx));
            let mul = _mm256_mul_ps(a_vec, b_vec);
            
            _mm256_storeu_ps(c.add(idx), mul);
        }
    }

    let remaining_start = chunks * 8;
    for i in remaining_start..len {
        *c.add(i) = a[i] * b[i]
    }
    
}
fn main() {

    let mat1 = Tensor::random(16, 28*28, 42);
    let vec1 = Tensor::random(28*28, 1, 24);

    let start = Instant::now();
    let res1 = mat1.mul_vec(&vec1);
    let simd_duration = start.elapsed();
    println!("SIMD duration: {:?}", simd_duration);

    let start = Instant::now();
    let res2 = mat1.mul_par(&vec1, 8);
    let mul_par_duration = start.elapsed();
    println!("mul_par duration: {:?}", mul_par_duration);

    let speedup = mul_par_duration.as_secs_f64() / simd_duration.as_secs_f64();
    println!("Speedup factor: {:.2}x", speedup);
    println!("res1 len: {}", res1.data.len());
    println!("res2 len: {}", res2.data.len());

}