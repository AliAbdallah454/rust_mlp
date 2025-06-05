use cp_proj::layer::ActivationType;
use cp_proj::mlp::MLP;
use cp_proj::mnits_data::MnistData;
use cp_proj::tensor::Tensor;
use std::fs::File;
use std::time::Instant;

fn main() {

    let train_data_result = MnistData::load_from_files(
        "./mnist/train-images.idx3-ubyte",
        "./mnist/train-labels.idx1-ubyte"
    );

    let data = train_data_result.unwrap();
    
    let mut training_images: Vec<Tensor> = Vec::with_capacity(data.images.len());
    let mut training_labels: Vec<Tensor> = Vec::with_capacity(data.labels.len());

    for image in data.images.iter() {
        training_images.push(Tensor::new(image.clone(), 28, 28));
    }
    
    for label in data.labels.iter() {
        let mut one_hot = vec![0.0; 10];
        one_hot[*label as usize] = 1.0;
        training_labels.push(Tensor::new(one_hot, 10, 1));
    }

    // ex1.print();

    // let layer_sizes = vec![1, 32, 32, 16, 1];
    // let activations = vec![ActivationType::Tanh, ActivationType::Tanh, ActivationType::Tanh, ActivationType::Linear];
    // let mut mlp = MLP::new(layer_sizes, activations, 0.01, 1, 42);

    // let num_samples = 100;
    // let mut inputs = Vec::with_capacity(num_samples);
    // let mut targets = Vec::with_capacity(num_samples);

    // for i in 0..num_samples {
    //     let x = 2.0 * std::f64::consts::PI * (i as f64) / (num_samples as f64);
    //     let y = x.sin();
    //     inputs.push(Tensor::scalar(x));
    //     targets.push(Tensor::scalar(y));
    // }

    // let start = std::time::Instant::now();
    // let _losses = mlp.train(&inputs, &targets, 600);
    // let duration = start.elapsed();

    // println!("Training took: {:.3?}", duration);

    // use std::fs::File;
    // use std::io::{Write, BufWriter};

    // let mut file = BufWriter::new(File::create("data.txt").expect("Unable to create file"));
    // for input in &inputs {
    //     let prediction = mlp.predict(input);
    //     writeln!(file, "{:.4},{:.4}", input.data[0], prediction.data[0]).expect("Unable to write data");
    // }

}