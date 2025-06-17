use cp_proj::helpers::split_dataset;
use cp_proj::layer::ActivationType;
use cp_proj::mlp::{LossFunction, MLP};
use cp_proj::mnits_data::MnistData;
use cp_proj::tensor::{ExecutionMode, Tensor};
use std::time::Instant;
use alloc::vec::Vec;


fn main() {
    let train_data_result = MnistData::load_from_files(
        "./mnist/train-images.idx3-ubyte",
        "./mnist/train-labels.idx1-ubyte"
    );

    let data = train_data_result.unwrap();
    
    let mut images: Vec<Tensor> = Vec::with_capacity(data.images.len());
    let mut labels: Vec<Tensor> = Vec::with_capacity(data.labels.len());

    for image in data.images.iter() {
        images.push(Tensor::new(image.clone(), 28*28, 1));
    }
    
    for label in data.labels.iter() {
        let mut one_hot = vec![0.0; 10];
        one_hot[*label as usize] = 1.0;
        labels.push(Tensor::new(one_hot, 10, 1));
    }

    // Use more images for better benchmarking
    images.truncate(1);
    labels.truncate(1);

    let (
        _training_images, 
        _training_labels, 
        testing_images, 
        testing_labels
    ) = split_dataset(images, labels, 0.8);

    let layer_sizes = vec![28*28, 1024*8, 1024*8, 10];
    let activations = vec![ActivationType::ReLU, ActivationType::ReLU, ActivationType::Softmax];
        
    let execution_modes = vec![
        ExecutionMode::Sequential, 
        ExecutionMode::Parallel, 
        ExecutionMode::SIMD, 
        ExecutionMode::ParallelSIMD
    ];
    
    println!("\nTesting inference performance across different execution modes ...");

    // Create and train a single MLP first
    let mut base_mlp = MLP::new(
        layer_sizes.clone(), 
        activations.clone(),
        LossFunction::CategoricalCrossEntropy, 
        0.01, 
        ExecutionMode::Sequential, 
        42
    );
    base_mlp.train(&testing_images, &testing_labels, 1);

    // Create MLPs with different execution modes using the same weights
    let mut mlps: Vec<MLP> = execution_modes.iter().map(|&execution_mode| {
        let mut mlp = MLP::new(
            layer_sizes.clone(), 
            activations.clone(),
            LossFunction::CategoricalCrossEntropy, 
            0.01, 
            execution_mode, 
            42
        );
        // Copy weights from base MLP
        mlp.layers = base_mlp.layers.clone();
        mlp
    }).collect();

    let mut durations = Vec::new();
    let num_runs = 5; // Number of times to run inference for averaging
    
    for (i, mlp) in mlps.iter_mut().enumerate() {
        let mut total_duration = std::time::Duration::new(0, 0);
        
        for _ in 0..num_runs {
            let inference_start = Instant::now();
            for image in &testing_images {
                let _prediction = mlp.predict(image);
            }
            total_duration += inference_start.elapsed();
        }
        
        let avg_duration = total_duration / num_runs as u32;
        durations.push(avg_duration);
        
        println!("{:?} inference time: {:.2?}", execution_modes[i], avg_duration);
    }

    // Calculate and print speedups relative to sequential execution
    let sequential_time = durations[0].as_secs_f64();
    println!("\nSpeedups relative to sequential execution:");
    for (i, duration) in durations.iter().enumerate() {
        let speedup = sequential_time / duration.as_secs_f64();
        println!("{:?}: {:.2}x", execution_modes[i], speedup);
    }
}