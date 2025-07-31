use rust_mlp::tensor::{ExecutionMode, Tensor};

fn main() {

    const M: usize = 64;
    const K: usize = 32;
    const N: usize = 128;

    let mat1 = Tensor::random_2d(M, K, 42);
    let mat2 = Tensor::random_2d(K, N, 24);

    let res_cpu = mat1.mul(&mat2, ExecutionMode::Sequential);
    let res = mat1.mul(&mat2, ExecutionMode::CuBLAS);

    println!("{}", res_cpu == res);

}