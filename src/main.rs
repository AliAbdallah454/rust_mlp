use cp_proj::helpers::{evaluate_model, split_dataset};
use cp_proj::layer::ActivationType;
use cp_proj::mlp::{LossFunction, MLP};
use cp_proj::mnits_data::MnistData;
use cp_proj::tensor::Tensor;
use std::time::Instant;

fn side_by_size(a: &Tensor, b: &Tensor) {
    assert_eq!(a.data.len(), b.data.len(), "Tensor data lengths must match");

    for i in 0..a.data.len() {
        println!("{} - {}: {}", a.data[i], b.data[i], a.data[i] == b.data[i]);
    }

}

fn main() {

    let mat1 = Tensor::random(16, 28*28, 42);
    let vec1 = Tensor::random(28*28, 1, 24);

    let start = Instant::now();
    let res1 = mat1.mul_vec(&vec1);
    let simd_duration = start.elapsed();
    println!("SIMD duration: {:?}", simd_duration);

    let start = Instant::now();
    let res2 = mat1.mul_par(&vec1, 1);
    let mul_par_duration = start.elapsed();
    println!("mul_par duration: {:?}", mul_par_duration);

    let speedup = mul_par_duration.as_secs_f64() / simd_duration.as_secs_f64();
    println!("Speedup factor: {:.2}x", speedup);
    println!("res1 len: {}", res1.data.len());
    println!("res2 len: {}", res2.data.len());

    // side_by_size(&res1, &res2);

    // res1.print();
    // println!("-----");
    // res2.print();

    // let train_data_result = MnistData::load_from_files(
    //     "./mnist/train-images.idx3-ubyte",
    //     "./mnist/train-labels.idx1-ubyte"
    // );

    // let data = train_data_result.unwrap();
    
    // let mut images: Vec<Tensor> = Vec::with_capacity(data.images.len());
    // let mut labels: Vec<Tensor> = Vec::with_capacity(data.labels.len());

    // for image in data.images.iter() {
    //     images.push(Tensor::new(image.clone(), 28*28, 1));
    // }
    
    // for label in data.labels.iter() {
    //     let mut one_hot = vec![0.0; 10];
    //     one_hot[*label as usize] = 1.0;
    //     labels.push(Tensor::new(one_hot, 10, 1));
    // }

    // // ----------------------------------------------

    // images = images.into_iter().skip(35_000).take(100).collect();
    // labels = labels.into_iter().skip(35_000).take(100).collect();

    // let mut mlp = MLP::load("./models/test3.txt").unwrap();

    // // Print header
    // println!("Predicted | Actual");
    // println!("------------------");

    // // Evaluate each image and print predictions
    // for (image, label) in images.iter().zip(labels.iter()) {
    //     let prediction = mlp.forward(image);
        
    //     // Get predicted class (index of max value)
    //     let pred_idx = prediction.data.iter()
    //         .enumerate()
    //         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //         .map(|(idx, _)| idx)
    //         .unwrap();
            
    //     // Get actual class (index of 1.0 in one-hot vector)
    //     let actual_idx = label.data.iter()
    //         .enumerate()
    //         .find(|(_, &val)| val == 1.0)
    //         .map(|(idx, _)| idx)
    //         .unwrap();
            
    //     println!("{:8} | {:6}", pred_idx, actual_idx);
    // }

    // let testing_accuracy = evaluate_model(&mut mlp, &images, &labels);
    // println!("\nOverall accuracy: {:.2}%", testing_accuracy * 100.0);

    // ----------------------------------------------

    // images.truncate(15000);
    // labels.truncate(15000);

    // let (
    //     training_images, 
    //     training_labels, 
    //     testing_images, 
    //     testing_labels
    // ) = split_dataset(images, labels, 0.8);

    // let layer_sizes = vec![28*28, 16, 16, 10];
    // let activations = vec![ActivationType::ReLU, ActivationType::ReLU, ActivationType::Softmax];

    // let mut mlp = MLP::new(layer_sizes, activations, LossFunction::CategoricalCrossEntropy, 0.01, false, 42);

    // println!("Accuracy before training:");
    // let training_accuracy = evaluate_model(&mut mlp, &training_images, &training_labels);
    // let testing_accuracy = evaluate_model(&mut mlp, &testing_images, &testing_labels);

    // println!("Training set: {:.2}%", training_accuracy * 100.0);
    // println!("Testing set: {:.2}%", testing_accuracy * 100.0);

    // println!("Training started ...");

    // let epochs = 1 as usize;
    // let train_start = Instant::now();
    // mlp.train(&training_images, &training_labels, epochs);
    // let train_duration = train_start.elapsed();
    // println!("Training for {} epochs completed in {:.2?}", epochs, train_duration);

    // let training_accuracy = evaluate_model(&mut mlp, &training_images, &training_labels);
    // let testing_accuracy = evaluate_model(&mut mlp, &testing_images, &testing_labels);

    // println!("Accuracy on training set: {:.2}%", training_accuracy * 100.0);
    // println!("Accuracy on testing set: {:.2}%", testing_accuracy * 100.0);

    // mlp.save("./models/test4_no_normalization.txt").unwrap();

}