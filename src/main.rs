use cp_proj::layer::ActivationType;
use cp_proj::mlp::{LossFunction, MLP};
use cp_proj::mnits_data::MnistData;
use cp_proj::tensor::Tensor;

fn main() {

    // let v = Tensor::new(vec![10.0, 1.0, 2.0, 3.0], 4, 1);
    // let v = v.softmax_derivative();
    // v.print();
    // v.sum();

    let train_data_result = MnistData::load_from_files(
        "./mnist/train-images.idx3-ubyte",
        "./mnist/train-labels.idx1-ubyte"
    );

    let data = train_data_result.unwrap();
    
    let mut training_images: Vec<Tensor> = Vec::with_capacity(data.images.len());
    let mut training_labels: Vec<Tensor> = Vec::with_capacity(data.labels.len());

    for image in data.images.iter() {
        training_images.push(Tensor::new(image.clone(), 28*28, 1));
    }
    
    for label in data.labels.iter() {
        let mut one_hot = vec![0.0; 10];
        one_hot[*label as usize] = 1.0;
        training_labels.push(Tensor::new(one_hot, 10, 1));
    }

    training_images.truncate(10000);
    training_labels.truncate(10000);

    let layer_sizes = vec![28*28, 16, 16, 10];
    let activations = vec![ActivationType::ReLU, ActivationType::ReLU, ActivationType::Linear];
    let mut mlp = MLP::new(layer_sizes, activations, LossFunction::MSE, 0.01, 1, 42);

    mlp.train(&training_images, &training_labels, 5);

}