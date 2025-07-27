use rust_mlp::helpers::split_dataset;
use rust_mlp::layer::ActivationType;
use rust_mlp::mlp::{LossFunction, MLP};
use rust_mlp::mnits_data::MnistData;
use rust_mlp::tensor::{ExecutionMode, Tensor};
use std::time::Instant;

fn main() {
    let train_data_result = MnistData::load_from_files(
        "./mnist/train-images.idx3-ubyte",
        "./mnist/train-labels.idx1-ubyte"
    );

    let data = train_data_result.unwrap();
    
    let mut images: Vec<Tensor> = Vec::with_capacity(data.images.len());
    let mut labels: Vec<Tensor> = Vec::with_capacity(data.labels.len());

    for image in data.images.iter() {
        images.push(Tensor::new_2d(image.clone(), 28*28, 1));
    }
    
    for label in data.labels.iter() {
        let mut one_hot = vec![0.0; 10];
        one_hot[*label as usize] = 1.0;
        labels.push(Tensor::new_2d(one_hot, 10, 1));
    }

    images.truncate(1);
    labels.truncate(1);

    let (
        training_images, 
        training_labels, 
        _testing_images, 
        _testing_labels
    ) = split_dataset(images, labels, 0.8);

    let layer_sizes = vec![28*28, 1048*8, 1048*8, 10];
    let activations = vec![ActivationType::ReLU, ActivationType::ReLU, ActivationType::Softmax];
        
    let execution_modes = vec![
        ExecutionMode::Sequential, 
        ExecutionMode::Parallel, 
        ExecutionMode::SIMD, 
        ExecutionMode::ParallelSIMD
    ];
    
    println!("\nTesting different Execution Modes ...");

    let mut mlps: Vec<MLP> = execution_modes.iter().map(|&execution_mode| {
        MLP::new(layer_sizes.clone(), activations.clone(), 
            LossFunction::CategoricalCrossEntropy, 0.01, execution_mode, 42)
    }).collect();

    let mut durations = Vec::new();
    
    for (i, mlp) in mlps.iter_mut().enumerate() {
        let train_start = Instant::now();
        mlp.train(&training_images, &training_labels, 1);
        let duration = train_start.elapsed();
        durations.push(duration);
        
        println!("{:?} mode training time: {:.2?}", execution_modes[i], duration);
    }

    let single_thread_time = durations[0].as_secs_f64();
    println!("\nSpeedups relative to Sequential Exectution:");
    for (i, duration) in durations.iter().enumerate() {
        let speedup = single_thread_time / duration.as_secs_f64();
        println!("{:?} mode: {:.2}x", execution_modes[i], speedup);
    }

}