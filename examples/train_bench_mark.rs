use cp_proj::helpers::split_dataset;
use cp_proj::layer::ActivationType;
use cp_proj::mlp::{LossFunction, MLP};
use cp_proj::mnits_data::MnistData;
use cp_proj::tensor::Tensor;
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
        images.push(Tensor::new(image.clone(), 28*28, 1));
    }
    
    for label in data.labels.iter() {
        let mut one_hot = vec![0.0; 10];
        one_hot[*label as usize] = 1.0;
        labels.push(Tensor::new(one_hot, 10, 1));
    }

    images.truncate(100);
    labels.truncate(100);

    let (
        training_images, 
        training_labels, 
        _testing_images, 
        _testing_labels
    ) = split_dataset(images, labels, 0.8);

    let layer_sizes = vec![28*28, 512, 512, 10];
    let activations = vec![ActivationType::ReLU, ActivationType::ReLU, ActivationType::Softmax];
    println!("\nTesting different thread counts (1, 4, 8) ...");
    
    let thread_counts = vec![1, 4, 8];
    let mut mlps: Vec<MLP> = thread_counts.iter().map(|&threads| {
        MLP::new(layer_sizes.clone(), activations.clone(), 
            LossFunction::CategoricalCrossEntropy, 0.01, threads, 42)
    }).collect();

    let mut durations = Vec::new();
    
    for (i, mlp) in mlps.iter_mut().enumerate() {
        let train_start = Instant::now();
        mlp.train(&training_images, &training_labels, 1);
        let duration = train_start.elapsed();
        durations.push(duration);
        
        println!("{} threads training time: {:.2?}", thread_counts[i], duration);
    }

    // Calculate and print speedups relative to single thread
    let single_thread_time = durations[0].as_secs_f64();
    println!("\nSpeedups relative to single thread:");
    for (i, duration) in durations.iter().enumerate() {
        let speedup = single_thread_time / duration.as_secs_f64();
        println!("{} threads: {:.2}x", thread_counts[i], speedup);
    }

}