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
    pub last_input: Option<Tensor>,
    pub last_pre_activation: Option<Tensor>,
    pub last_output: Option<Tensor>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType, seed: u64) -> Self {

        // Xavier Initialization
        let scale = (2.0 / input_size as f32).sqrt();
        let mut weights = Tensor::random(output_size, input_size, seed);
        weights = weights.scale(scale);
        
        let biases = Tensor::zeros(output_size, 1);
        
        Self {
            weights,
            biases,
            activation,
            last_input: None,
            last_pre_activation: None,
            last_output: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor, nb_threads: usize) -> Tensor {

        self.last_input = Some(input.clone());
        
        // Compute z = W * x + b
        let z = self.weights.mul_par(input, nb_threads);
        let z_with_bias = &z + &self.biases;
        
        self.last_pre_activation = Some(z_with_bias.clone());
        
        let output = match self.activation {
            ActivationType::ReLU => z_with_bias.relu(),
            ActivationType::Sigmoid => z_with_bias.sigmoid(),
            ActivationType::Linear => z_with_bias,
            ActivationType::Tanh => z_with_bias.tanh(),
            ActivationType::Softmax => z_with_bias.softmax()
        };
        
        self.last_output = Some(output.clone());
        
        output
    }

    pub fn backward(&self, gradient: &Tensor, nb_threads: usize) -> (Tensor, Tensor, Tensor) {

        let input = self.last_input.as_ref().expect("Forward pass must be called before backward");
        let pre_activation = self.last_pre_activation.as_ref().expect("Forward pass must be called before backward");
        
        let activation_derivative = match self.activation {
            ActivationType::ReLU => pre_activation.relu_derivative(),
            ActivationType::Sigmoid => pre_activation.sigmoid_derivative(),
            ActivationType::Linear => Tensor::ones(pre_activation.rows, pre_activation.cols),
            ActivationType::Tanh => pre_activation.tanh_derivative(),
            ActivationType::Softmax => pre_activation.softmax_derivative()
        };
        

        // For Softmax, the derivative is a Jacobian, so we need to do a matrix-vector product
        // dL/dz = dL/da * da/dz (we have dL/da and da/dz)
        let dz = match self.activation {
            ActivationType::Softmax => {
                activation_derivative.mul_par(gradient, nb_threads)
            },
            _ => {
                gradient.hadamard(&activation_derivative)
            }
        }.clone();

        // Gradient w.r.t. weights: dL/dW = dL/dz * x^T
        let input_t = input.transpose();
        let dw = dz.mul_par(&input_t, nb_threads);
        
        // Gradient w.r.t. biases: dL/db = dL/dz
        let db = dz.clone();
        
        // Gradient w.r.t. input: dL/dx = W^T * dL/dz
        let weights_t = self.weights.transpose();
        let dx = weights_t.mul_par(&dz, nb_threads);
        
        (dx, dw, db)
    }

    pub fn update_weights(&mut self, dw: &Tensor, db: &Tensor, learning_rate: f32) {

        // Update weights: W = W - lr * dW
        let weight_update = dw.scale(learning_rate);
        self.weights = &self.weights - &weight_update;
        
        // Update biases: b = b - lr * db
        let bias_update = db.scale(learning_rate);
        self.biases = &self.biases - &bias_update;
    }
}