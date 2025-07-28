use crate::{activation_functions::activation_function::ActivationFunction, mlp::LossFunctionEnum, tensor::{ExecutionMode, Tensor}};
use crate::activation_functions::{Relu, Sigmoid, Tanh, Softmax};

#[derive(Clone, Debug,PartialEq)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Linear,
    Tanh,
    Softmax
}

#[derive(Clone)]
pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: ActivationType,
    pub last_input: Option<Tensor>,
    pub last_pre_activation: Option<Tensor>,
    pub last_output: Option<Tensor>,
    pub loss_function: LossFunctionEnum
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType, loss_functions: LossFunctionEnum, seed: u64) -> Self {

        // Xavier Initialization
        let scale = (2.0 / input_size as f32).sqrt();
        let mut weights = Tensor::random_2d(output_size, input_size, seed);
        weights = weights.scale(scale);
        
        let biases = Tensor::zeros_2d(output_size, 1);
        
        Self {
            weights,
            biases,
            activation,
            last_input: None,
            last_pre_activation: None,
            last_output: None,
            loss_function: loss_functions
        }
    }

    pub fn forward(&mut self, input: &Tensor, execution_mode: ExecutionMode) -> Tensor {

        self.last_input = Some(input.clone());
        
        // Compute z = W * x + b
        let z = self.weights.mul(input, execution_mode);
        let z_with_bias = &z + &self.biases;
        
        self.last_pre_activation = Some(z_with_bias.clone());
        
        let output = match self.activation {
            ActivationType::ReLU => Relu::forward(&z_with_bias),
            ActivationType::Sigmoid => Sigmoid::forward(&z_with_bias),
            ActivationType::Linear => z_with_bias,
            ActivationType::Tanh => Tanh::forward(&z_with_bias),
            ActivationType::Softmax => Softmax::forward(&z_with_bias)
        };

        self.last_output = Some(output.clone());
        
        output
    }

    pub fn backward(&self, gradient: &Tensor, execution_mode: ExecutionMode) -> (Tensor, Tensor, Tensor) {

        let input = self.last_input.as_ref().expect("Forward pass must be called before backward");
        let pre_activation = self.last_pre_activation.as_ref().expect("Forward pass must be called before backward");
        
        let activation_derivative = match self.activation {
            ActivationType::ReLU => Relu::derivative(pre_activation),
            ActivationType::Sigmoid => Sigmoid::derivative(pre_activation),
            ActivationType::Linear => Tensor::ones(pre_activation.shape.clone()),
            ActivationType::Tanh => Tanh::derivative(pre_activation),
            
            // For softmax activation function, this is useless since the drivative of the cross entropy takes into considiration the 
            // derivative of the softmax.
            ActivationType::Softmax => Tensor::zeros_2d(pre_activation.rows(), pre_activation.cols())
        };

        // For Softmax, the derivative is a Jacobian, so we need to do a matrix-vector product
        // dL/dz = dL/da * da/dz (we have dL/da and da/dz)
        let dz = match self.activation {
            ActivationType::Softmax => {
                gradient.clone()
            },
            _ => {
                gradient.hadamard(&activation_derivative)
            }
        }.clone();

        // Gradient w.r.t weights: dL/dW = dL/dz * x^T
        let input_t = input.transpose();
        let dw = dz.mul(&input_t, execution_mode);
        
        // Gradient w.r.t biases: dL/db = dL/dz
        let db = dz.clone();
        
        // Gradient w.r.t input: dL/dx = W^T * dL/dz
        let weights_t = self.weights.transpose();
        let dx = weights_t.mul(&dz, execution_mode);
        
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