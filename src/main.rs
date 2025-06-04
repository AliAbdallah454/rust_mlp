use std::error;

use cp_proj::layer::Layer;
use cp_proj::layer::ActivationType;
use cp_proj::mlp::{self, MLP};
use cp_proj::tensor::Tensor;

fn main() {

    // let data = vec![1.0, 1.0];
    // let expect = Tensor::scalar(1.0);

    // let input = Tensor::new(data, 2, 1);

    // let layer1 = Layer::new(2, 2, 42);
    // let layer2 = Layer::new(1, 2, 42);

    // let layers = vec![layer1, layer2];
    // let mlp = MLP::new(layers, 1);

    // let (res, err) = mlp.train_iter(input.clone(), expect);
    
    // res.print();
    // err.print();
    // let (r, c) = res.dims();
    // println!("output dim: ({}, {})", r, c);

    let layer_sizes = vec![2, 2, 1];
    let activations = vec![ActivationType::ReLU, ActivationType::Sigmoid];
    let mut mlp = MLP::new(layer_sizes, activations, 0.05, 4, 50);

    let inputs = vec![
        Tensor::new(vec![0.0, 0.0], 2, 1),
        Tensor::new(vec![0.0, 1.0], 2, 1),
        Tensor::new(vec![1.0, 0.0], 2, 1),
        Tensor::new(vec![1.0, 1.0], 2, 1),
    ];

    let targets = vec![
        Tensor::new(vec![1.0], 1, 1),
        Tensor::new(vec![0.0], 1, 1),
        Tensor::new(vec![0.0], 1, 1),
        Tensor::new(vec![1.0], 1, 1),
    ];

    let start = std::time::Instant::now();
    let losses = mlp.train(&inputs, &targets, 1000);
    let duration = start.elapsed();

    println!("Training took: {:.3?}", duration);

    // Test predictions
    for (i, input) in inputs.iter().enumerate() {
        let prediction = mlp.predict(input);
        println!("Input: {:?}, Target: {:.1}, Prediction: {:.3}", 
                    &input.data, targets[i].data[0], prediction.data[0]);
    }

}