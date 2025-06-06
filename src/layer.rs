use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Linear,
    Tanh,
    Softmax
}

pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: ActivationType,
    pub concurrent: bool,
    pub last_input: Option<Tensor>,
    pub last_pre_activation: Option<Tensor>,
    pub last_output: Option<Tensor>,
}

impl Layer {
    pub fn new(input_size: u32, output_size: u32, activation: ActivationType, concurrent: bool, seed: u64) -> Self {

        // Xavier Initialization
        let scale = (2.0 / input_size as f64).sqrt();
        let mut weights = Tensor::random(output_size, input_size, seed);
        weights.set_concurrent(concurrent);

        weights = weights.scale(scale);
        
        let mut biases = Tensor::zeros(output_size, 1);        
        biases.set_concurrent(concurrent);

        Self {
            weights,
            biases,
            activation,
            concurrent,
            last_input: None,
            last_pre_activation: None,
            last_output: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {

        let mut t = input.clone();
        t.set_concurrent(self.concurrent);
        self.last_input = Some(t);
        
        // Compute z = W * x + b
        let z = self.weights.mul(input);
        let mut z_with_bias = &z + &self.biases;
        z_with_bias.set_concurrent(self.concurrent);

        self.last_pre_activation = Some(z_with_bias.clone());
        
        let mut output = match self.activation {
            ActivationType::ReLU => z_with_bias.relu(),
            ActivationType::Sigmoid => z_with_bias.sigmoid(),
            ActivationType::Linear => z_with_bias,
            ActivationType::Tanh => z_with_bias.tanh(),
            ActivationType::Softmax => z_with_bias.softmax()
        };
        output.set_concurrent(self.concurrent);

        self.last_output = Some(output.clone());
        
        output
    }

    pub fn backward(&self, gradient: &Tensor) -> (Tensor, Tensor, Tensor) {

        let input = self.last_input.as_ref().expect("Forward pass must be called before backward");
        let pre_activation = self.last_pre_activation.as_ref().expect("Forward pass must be called before backward");
        
        let mut activation_derivative = match self.activation {
            ActivationType::ReLU => pre_activation.relu_derivative(),
            ActivationType::Sigmoid => pre_activation.sigmoid_derivative(),
            ActivationType::Linear => Tensor::ones(pre_activation.rows, pre_activation.cols),
            ActivationType::Tanh => pre_activation.tanh_derivative(),
            ActivationType::Softmax => pre_activation.softmax_derivative()
        };
        activation_derivative.set_concurrent(self.concurrent);

        // For Softmax, the derivative is a Jacobian, so we need to do a matrix-vector product
        // dL/dz = dL/da * da/dz (we have dL/da and da/dz)
        let mut dz = match self.activation {
            ActivationType::Softmax => {
                activation_derivative.mul(gradient)
            },
            _ => {
                gradient.hadamard(&activation_derivative)
            }
        }.clone();
        dz.set_concurrent(self.concurrent);

        // Gradient w.r.t. weights: dL/dW = dL/dz * x^T
        let mut input_t = input.transpose();
        input_t.set_concurrent(self.concurrent);
        let mut dw = dz.mul(&input_t);
        dw.set_concurrent(self.concurrent);

        // Gradient w.r.t. biases: dL/db = dL/dz
        let mut db = dz.clone();
        db.set_concurrent(self.concurrent);

        // Gradient w.r.t. input: dL/dx = W^T * dL/dz
        let mut weights_t = self.weights.transpose();
        weights_t.set_concurrent(self.concurrent);

        let mut dx = weights_t.mul(&dz);
        dx.set_concurrent(self.concurrent);

        (dx, dw, db)
    }

    pub fn update_weights(&mut self, dw: &Tensor, db: &Tensor, learning_rate: f64) {

        // Update weights: W = W - lr * dW
        let mut weight_update = dw.scale(learning_rate);
        weight_update.set_concurrent(self.concurrent);

        self.weights = &self.weights - &weight_update;
        
        // Update biases: b = b - lr * db
        let mut bias_update = db.scale(learning_rate);
        bias_update.set_concurrent(self.concurrent);

        self.biases = &self.biases - &bias_update;
    }
}