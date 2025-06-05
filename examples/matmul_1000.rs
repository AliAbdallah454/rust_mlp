use cp_proj::Tensor;

fn main() {

    let mat1 = Tensor::random(1000, 11000, 42);
    let mat2 = Tensor::random(11000, 900, 42);

    let start_par = std::time::Instant::now();
    let res_par = mat1.mul_par(&mat2, 4);
    let duration_par = start_par.elapsed();
    println!("mul_par (4 threads) took: {:.3?}", duration_par);

    let start_seq = std::time::Instant::now();
    let res_seq = mat1.mul_seq(&mat2);
    let duration_seq = start_seq.elapsed();
    println!("mul_seq took: {:.3?}", duration_seq);

    let start_simd = std::time::Instant::now();
    let res_simd = mat1.mul_par_simd(&mat2, 4);
    let duration_simd = start_simd.elapsed();
    println!("mul_par_simd (4 threads) took: {:.3?}", duration_simd);

}