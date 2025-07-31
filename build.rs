use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {

    println!("cargo:rerun-if-changed=cuda/mat_mul_kernel.cu");
    println!("cargo:rerun-if-changed=cuda/cuBLAS_mat_mul_kernel.cu");
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(&out_dir);
    
    // Compile regular mat_mul_kernel.cu to .o
    let obj_file = out_path.join("mat_mul_kernel.o");
    let status = Command::new("nvcc")
        .args(&[
            "-c", "cuda/mat_mul_kernel.cu",
            "-o", obj_file.to_str().unwrap(),
            "-Xcompiler", "-fPIC",
        ])
        .status()
        .expect("Failed to run nvcc. Make sure CUDA is installed and nvcc is in PATH");

    if !status.success() {
        panic!("nvcc compilation failed");
    }

    // Compile cuBLAS kernel to .o
    let cublas_obj_file = out_path.join("cuBLAS_mat_mul_kernel.o");
    let status = Command::new("nvcc")
        .args(&[
            "-c", "cuda/cuBLAS_mat_mul_kernel.cu",
            "-o", cublas_obj_file.to_str().unwrap(),
            "-Xcompiler", "-fPIC",
        ])
        .status()
        .expect("Failed to run nvcc for cuBLAS kernel");

    if !status.success() {
        panic!("nvcc compilation failed for cuBLAS kernel");
    }

    // Archive both .o files into a .a (static library)
    let lib_file = out_path.join("libmat_mul_cuda.a");
    let status = Command::new("ar")
        .args(&["rcs", lib_file.to_str().unwrap(), 
                obj_file.to_str().unwrap(),
                cublas_obj_file.to_str().unwrap()])
        .status()
        .expect("Failed to run ar");

    if !status.success() {
        panic!("ar archiving failed");
    }

    // Tell Cargo where to find the library and link it
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=mat_mul_cuda"); // mat_mul_cuda translates to -> libmat_mul_cuda.a
    
    // Link CUDA runtime and cuBLAS
    println!("cargo:rustc-link-lib=cudart"); // translates to -> libcudart.so (.so => dynamically linked)
    println!("cargo:rustc-link-lib=cublas"); // translates to -> libcublas.so (.so => dynamically linked)
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64"); // where to look for these dyn linked libraries
}
