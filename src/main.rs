use cp_proj::helpers::{evaluate_model, split_dataset};
use cp_proj::layer::ActivationType;
use cp_proj::mlp::{LossFunction, MLP};
use cp_proj::mnits_data::MnistData;
use cp_proj::tensor::Tensor;
use std::time::Instant;

fn main() {

    // let mut mat1 = Tensor::random(1000, 1000, 42);
    // let mat2 = Tensor::random(1000, 1000, 43);
    // let start1 = Instant::now();
    // mat1.mul_par_old(&mat2, 12);
    // let duration1 = start1.elapsed();
    // println!("threaded_mul took: {:?}", duration1);

    // let start2 = Instant::now();
    // mat1.mul(&mat2);
    // let duration2 = start2.elapsed();
    // println!("sequential_mul took: {:?}", duration2);

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

    // // ----------------------------------------------

    // // images = images.into_iter().skip(35_000).take(100).collect();
    // // labels = labels.into_iter().skip(35_000).take(100).collect();

    // // images = images.into_iter().skip(15_000).collect();
    // // labels = labels.into_iter().skip(15_000).collect();


    // // let mut mlp = MLP::load("./models/test3.txt").unwrap();

    // // let testing_accuracy = evaluate_model(&mut mlp, &images, &labels);
    // // println!("Testing set: {:.2}%", testing_accuracy * 100.0);

    // // ----------------------------------------------

    images.truncate(15000);
    labels.truncate(15000);

    let (
        training_images, 
        training_labels, 
        testing_images, 
        testing_labels
    ) = split_dataset(images, labels, 0.8);

    let layer_sizes = vec![28*28, 16, 16, 10];
    let activations = vec![ActivationType::ReLU, ActivationType::ReLU, ActivationType::Softmax];

    let mut mlp = MLP::new(layer_sizes, activations, LossFunction::CategoricalCrossEntropy, 0.01, false, 42);

    println!("Accuracy before training:");
    let training_accuracy = evaluate_model(&mut mlp, &training_images, &training_labels);
    let testing_accuracy = evaluate_model(&mut mlp, &testing_images, &testing_labels);

    println!("Training set: {:.2}%", training_accuracy * 100.0);
    println!("Testing set: {:.2}%", testing_accuracy * 100.0);

    println!("Training started ...");

    let epochs = 1 as usize;
    let train_start = Instant::now();
    mlp.train(&training_images, &training_labels, epochs);
    let train_duration = train_start.elapsed();
    println!("Training for {} epochs completed in {:.2?}", epochs, train_duration);

    let training_accuracy = evaluate_model(&mut mlp, &training_images, &training_labels);
    let testing_accuracy = evaluate_model(&mut mlp, &testing_images, &testing_labels);

    println!("Accuracy on training set: {:.2}%", training_accuracy * 100.0);
    println!("Accuracy on testing set: {:.2}%", testing_accuracy * 100.0);

    // mlp.save("./models/test4_no_normalization.txt").unwrap();

}