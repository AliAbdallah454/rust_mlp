use cp_proj::layer::Layer;
use cp_proj::mlp::MLP;
use cp_proj::tensor::Tensor;

fn main() {

    let flattened_image = Tensor::random(28*28, 1, 42);

    let layer1 = Layer::new(256, 28*28, 42);
    let layer2 = Layer::new(16, 256, 42);
    let layer3 = Layer::new(1, 16, 42);

    let layers = vec![layer1, layer2, layer3];

    let mlp = MLP::new(layers, 4);
    let res = mlp.forward(flattened_image);

    res.print();
    let (r, c) = res.dims();
    println!("output dim: ({}, {})", r, c);

}