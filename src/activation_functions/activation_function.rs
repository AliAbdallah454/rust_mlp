use crate::tensor::Tensor;

pub trait ActivationFunction {

    fn forward(layer: &Tensor) -> Tensor;
    fn derivative(layer: &Tensor) -> Tensor;

}

pub struct Relu;

impl ActivationFunction for Relu {

    fn forward(layer: &Tensor) -> Tensor {
        let data = layer.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        Tensor::new(data, layer.shape.clone())
    }

    fn derivative(layer: &Tensor) -> Tensor {
        let data = layer.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        Tensor::new(data, layer.shape.clone())
    }

}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {

    fn forward(layer: &Tensor) -> Tensor {
        let data = layer.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::new(data, layer.shape.clone())
    }

    /// Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    fn derivative(layer: &Tensor) -> Tensor {
        let sigmoid_vals = Sigmoid::forward(layer);
        let ones = Tensor::ones(layer.shape.clone());
        let one_minus_sigmoid = &ones - &sigmoid_vals;
        sigmoid_vals.hadamard(&one_minus_sigmoid)
    }

}

pub struct Tanh;

impl ActivationFunction for Tanh {
    fn forward(layer: &Tensor) -> Tensor {
        let data = layer.data.iter().map(|&x| x.tanh()).collect();
        Tensor::new(data, layer.shape.clone())
    }

    /// Tanh derivative: 1 - tanh(x)^2
    fn derivative(layer: &Tensor) -> Tensor {
        let tanh_vals = Tanh::forward(layer);
        let ones = Tensor::ones(layer.shape.clone());
        let tanh_squared = tanh_vals.hadamard(&tanh_vals);
        &ones - &tanh_squared
    }
}

pub struct Softmax;

impl ActivationFunction for Softmax {
    fn forward(layer: &Tensor) -> Tensor {
        assert_eq!(layer.cols(), 1, "Softmax only implemented for column vectors (r x 1)");

        let exp_vals: Vec<f32> = layer.data.iter().map(|&x| x.exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let data: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();
        Tensor::new(data, layer.shape.clone())
    }

    // to get the derivative of the softmax we need to calculate its Jacobian Matrix
    fn derivative(layer: &Tensor) -> Tensor {
        assert_eq!(layer.cols(), 1, "Softmax derivative only implemented for column vectors (r x 1)");
    
        // let softmax = self.softmax();
        let softmax = Softmax::forward(layer);
        let len = softmax.data.len();
        let mut jacobian = vec![0.0; len * len];
    
        for i in 0..len {
            for j in 0..len {
                let s_i = softmax.data[i];
                let s_j = softmax.data[j];
                let val = if i == j {
                    s_i * (1.0 - s_i)
                } else {
                    -s_i * s_j
                };
                jacobian[i * len + j] = val;
            }
        }
    
        Tensor::new(jacobian, vec![len, len]) // Jacobian is (r x r)
    }
}