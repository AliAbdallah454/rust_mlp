use cp_proj::layer::ActivationType;
use cp_proj::mlp::MLP;
use cp_proj::tensor::Tensor;

fn main() {

    let layer_sizes = vec![1, 32, 32, 16, 1];
    let activations = vec![ActivationType::Tanh, ActivationType::Tanh, ActivationType::Tanh, ActivationType::Linear];
    let mut mlp = MLP::new(layer_sizes, activations, 0.01, 1, 42);

    let num_samples = 100;
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let x = 2.0 * std::f64::consts::PI * (i as f64) / (num_samples as f64);
        let y = x.sin();
        inputs.push(Tensor::new(vec![x], 1, 1));
        targets.push(Tensor::new(vec![y], 1, 1));
    }

    let start = std::time::Instant::now();
    let _losses = mlp.train(&inputs, &targets, 500);
    let duration = start.elapsed();

    println!("Training took: {:.3?}", duration);

    use std::fs::File;
    use std::io::{Write, BufWriter};

    let mut file = BufWriter::new(File::create("data.txt").expect("Unable to create file"));
    for input in &inputs {
        let prediction = mlp.predict(input);
        writeln!(file, "{:.4},{:.4}", input.data[0], prediction.data[0]).expect("Unable to write data");
    }

}