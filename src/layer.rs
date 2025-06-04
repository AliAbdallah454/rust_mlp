// use crate::tensor::Tensor;

// pub struct Layer {
//     pub weights: Tensor,
// }

// impl Layer {
//     pub fn new(rows: u32, cols: u32, seed: u64) -> Self {
//         Self { weights: Tensor::random(rows, cols, seed) }
//     }

//     pub fn forward(&self, input: &Tensor, nb_threads: usize) -> Tensor {
//         self.weights.mul_par(input, nb_threads)
//     }
    
// }


use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Linear,
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
    pub fn new(input_size: u32, output_size: u32, activation: ActivationType, seed: u64) -> Self {
        // Xavier initialization
        let scale = (2.0 / input_size as f64).sqrt();
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
        // Store input for backpropagation
        self.last_input = Some(input.clone());
        
        // Compute z = W * x + b
        let z = self.weights.mul_par(input, nb_threads);
        let z_with_bias = &z + &self.biases;
        
        // Store pre-activation for backpropagation
        self.last_pre_activation = Some(z_with_bias.clone());
        
        // Apply activation function
        let output = match self.activation {
            ActivationType::ReLU => z_with_bias.relu(),
            ActivationType::Sigmoid => z_with_bias.sigmoid(),
            ActivationType::Linear => z_with_bias,
        };
        
        // Store output for backpropagation
        self.last_output = Some(output.clone());
        
        output
    }

    pub fn backward(&self, gradient: &Tensor) -> (Tensor, Tensor, Tensor) {
        let input = self.last_input.as_ref().expect("Forward pass must be called before backward");
        let pre_activation = self.last_pre_activation.as_ref().expect("Forward pass must be called before backward");
        
        // Compute activation derivative
        let activation_derivative = match self.activation {
            ActivationType::ReLU => pre_activation.relu_derivative(),
            ActivationType::Sigmoid => pre_activation.sigmoid_derivative(),
            ActivationType::Linear => Tensor::ones(pre_activation.rows, pre_activation.cols),
        };
        
        // Gradient w.r.t. pre-activation: dL/dz = dL/da * da/dz
        let dz = gradient.hadamard(&activation_derivative);
        
        // Gradient w.r.t. weights: dL/dW = dL/dz * x^T
        let input_t = input.transpose();
        let dw = dz.mul_par(&input_t, 1);
        
        // Gradient w.r.t. biases: dL/db = dL/dz (sum over batch dimension if needed)
        let db = dz.clone();
        
        // Gradient w.r.t. input: dL/dx = W^T * dL/dz
        let weights_t = self.weights.transpose();
        let dx = weights_t.mul_par(&dz, 1);
        
        (dx, dw, db)
    }

    pub fn update_weights(&mut self, dw: &Tensor, db: &Tensor, learning_rate: f64) {
        // Update weights: W = W - lr * dW
        let weight_update = dw.scale(learning_rate);
        self.weights = &self.weights - &weight_update;
        
        // Update biases: b = b - lr * db
        let bias_update = db.scale(learning_rate);
        self.biases = &self.biases - &bias_update;
    }
}
