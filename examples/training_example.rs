use rust_mlp::helpers::{evaluate_model, split_dataset};
use rust_mlp::layer::ActivationType;
use rust_mlp::mlp::{LossFunction, MLP};
use rust_mlp::mnits_data::MnistData;
use rust_mlp::tensor::{ExecutionMode, Tensor};

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

    images.truncate(1_000);
    labels.truncate(1_000);

    let (
        training_images, 
        training_labels, 
        testing_images, 
        testing_labels
    ) = split_dataset(images, labels, 0.8);

    let layer_sizes = vec![28*28, 16, 16, 10];
    let activations = vec![ActivationType::ReLU, ActivationType::ReLU, ActivationType::Softmax];
        
    let mut mlp = MLP::new(layer_sizes, activations, LossFunction::CategoricalCrossEntropy, 0.05, ExecutionMode::ParallelSIMD, 42);

    mlp.train(&training_images, &training_labels, 17);

    let accuracy = evaluate_model(&mut mlp, &testing_images, &testing_labels);
    println!("Accuracy is: {}", accuracy);

    mlp.save("./models/api2-test2.txt").unwrap();

}