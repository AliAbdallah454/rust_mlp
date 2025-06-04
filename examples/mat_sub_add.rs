// use cp_proj::Tensor;

// fn main() {

//     let mat1 = Tensor::random(1000, 1000, 42);
//     let mat2 = Tensor::random(1000, 1000, 43);

//     let start_seq = std::time::Instant::now();
//     mat1.add_par(&mat2, 1);
//     let duration_seq = start_seq.elapsed();
//     println!("add_par (1 thread) took: {:.3?}", duration_seq);

//     let start_par = std::time::Instant::now();
//     mat1.add_par(&mat2, 4);
//     let duration_par = start_par.elapsed();
//     println!("add_par (4 threads) took: {:.3?}", duration_par);

// }