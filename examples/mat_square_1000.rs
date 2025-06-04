// use cp_proj::Tensor;

// fn main() {

//     let mat = Tensor::random(1000, 1000, 42);

//     let start_seq = std::time::Instant::now();
//     mat.square_par(1);
//     let duration_seq = start_seq.elapsed();
//     println!("square_par (1 thread) took: {:.3?}", duration_seq);

//     let start_par = std::time::Instant::now();
//     mat.square_par(4);
//     let duration_par = start_par.elapsed();
//     println!("square_par (4 threads) took: {:.3?}", duration_par);

// }