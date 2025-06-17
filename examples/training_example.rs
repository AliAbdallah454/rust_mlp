use cp_proj::helpers::{evaluate_model, split_dataset};
use cp_proj::layer::ActivationType;
use cp_proj::mlp::{LossFunction, MLP};
use cp_proj::mnits_data::MnistData;
use cp_proj::tensor::{ExecutionMode, Tensor};
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

    images.truncate(15_000);
    labels.truncate(15_000);

    let (
        training_images, 
        training_labels, 
        testing_images, 
        testing_labels
    ) = split_dataset(images, labels, 0.8);

    let layer_sizes = vec![28*28, 16, 16, 10];
    let activations = vec![ActivationType::ReLU, ActivationType::ReLU, ActivationType::Softmax];
        
    println!("\nTesting different Execution Modes ...");

    let mut mlp = MLP::new(layer_sizes, activations, LossFunction::CategoricalCrossEntropy, 0.05, ExecutionMode::ParallelSIMD, 42);

    mlp.train(&training_images, &training_labels, 10);

    let accuracy = evaluate_model(&mut mlp, &testing_images, &testing_labels);
    println!("Accuracy is: {}", accuracy);

    mlp.save("./models/api2-test1.txt").unwrap();

}