use crate::layer::{Layer, ActivationType};
use crate::loss_functions::loss_function::LossFunction;
use crate::loss_functions::{CategoricalCrossEntropy, MSE};
use crate::tensor::{ExecutionMode, Tensor};

use std::fs::File;
use std::io::{Write, Read, BufWriter, BufReader};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum LossFunctionEnum {
    MSE,
    CategoricalCrossEntropy
}

pub struct MLP {
    pub layers: Vec<Layer>,
    pub execution_mode: ExecutionMode,
    pub learning_rate: f32,
    pub loss_function: LossFunctionEnum 
}

impl MLP {
    pub fn new(layer_sizes: Vec<usize>, activations: Vec<ActivationType>, loss_function: LossFunctionEnum, learning_rate: f32, execution_mode: ExecutionMode, seed: u64) -> Self {
        assert_eq!(layer_sizes.len() - 1, activations.len(), "Number of activations must match number of layers");
        
        let mut layers = Vec::new();
        let mut current_seed = seed;
        
        for i in 0..layer_sizes.len() - 1 {
            let layer = Layer::new(
                layer_sizes[i], 
                layer_sizes[i + 1], 
                activations[i].clone(),
                loss_function.clone(),
                current_seed
            );
            layers.push(layer);
            current_seed = current_seed.wrapping_add(1); // Different seed for each layer
        }
        
        MLP {
            layers,
            execution_mode,
            learning_rate,
            loss_function,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut current_input = input.clone();
        
        for (_i, layer) in &mut self.layers.iter_mut().enumerate() {
            current_input = layer.forward(&current_input, self.execution_mode);
        }
        current_input
    }

    pub fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> f32 {
        
        // Compute loss
        let (loss, mut gradient) = match self.loss_function {
            LossFunctionEnum::MSE => (
                // prediction.mse_loss(target),
                // prediction.mse_loss_derivative(target)
                MSE::forward(prediction, target),
                MSE::backward(prediction, target)
            ),
            LossFunctionEnum::CategoricalCrossEntropy => (
                // prediction.categorical_cross_entropy(target),
                // prediction.categorical_cross_entropy_derivative(target)
                CategoricalCrossEntropy::forward(prediction, target),
                CategoricalCrossEntropy::backward(prediction, target)
            ),
        };

        // Backpropagate through layers in reverse order
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();
        
        for layer in self.layers.iter().rev() {
            let (dx, dw, db) = layer.backward(&gradient, self.execution_mode);
            weight_gradients.push(dw);
            bias_gradients.push(db);
            gradient = dx;
        }
        
        // Reverse gradients to match layer order
        weight_gradients.reverse();
        bias_gradients.reverse();
        
        // Update weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.update_weights(&weight_gradients[i], &bias_gradients[i], self.learning_rate);
        }
        
        loss
    }

    pub fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32 {
        let prediction = self.forward(input);
        self.backward(&prediction, target)
    }

    pub fn train(&mut self, inputs: &[Tensor], targets: &[Tensor], epochs: usize) -> Vec<f32> {
        assert_eq!(inputs.len(), targets.len(), "Number of inputs must match number of targets");
        
        let mut losses = Vec::new();
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let loss = self.train_step(input, target);
                epoch_loss += loss;
            }
            
            epoch_loss /= inputs.len() as f32;
            losses.push(epoch_loss);
            
            println!("Epoch {}: Loss = {:.6}", epoch, epoch_loss);
        }
        
        losses
    }

    pub fn predict(&mut self, input: &Tensor) -> Tensor {
        self.forward(input)
    }
}
impl MLP {
    /// Save the entire MLP (weights, biases, architecture, hyperparameters) to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write MLP metadata
        writeln!(writer, "MLP_SAVE_FORMAT_V1")?;
        writeln!(writer, "{}", self.layers.len())?;
        
        // Write execution mode
        match self.execution_mode {
            ExecutionMode::Sequential => writeln!(writer, "Sequential")?,
            ExecutionMode::Parallel => writeln!(writer, "Parallel")?,
            ExecutionMode::SIMD => writeln!(writer, "SIMD")?,
            ExecutionMode::ParallelSIMD => writeln!(writer, "ParallelSIMD")?,
            ExecutionMode::Cuda => writeln!(writer, "Cuda")?,
            ExecutionMode::CuBLAS => writeln!(writer, "CuBLAS")?,
        }
        
        writeln!(writer, "{:.17}", self.learning_rate)?; // High precision for f32
        
        // Write loss function
        match self.loss_function {
            LossFunctionEnum::MSE => writeln!(writer, "MSE")?,
            LossFunctionEnum::CategoricalCrossEntropy => writeln!(writer, "CategoricalCrossEntropy")?,
        }
        
        // Write each layer
        for layer in &self.layers {
            // Write layer metadata
            writeln!(writer, "{} {}", layer.weights.rows(), layer.weights.cols())?;
            
            // Write activation type
            match layer.activation {
                ActivationType::ReLU => writeln!(writer, "ReLU")?,
                ActivationType::Sigmoid => writeln!(writer, "Sigmoid")?,
                ActivationType::Linear => writeln!(writer, "Linear")?,
                ActivationType::Tanh => writeln!(writer, "Tanh")?,
                ActivationType::Softmax => writeln!(writer, "Softmax")?,
            }
            
            // Write weights
            for &weight in &layer.weights.data {
                writeln!(writer, "{:.17}", weight)?;
            }
            
            // Write biases
            for &bias in &layer.biases.data {
                writeln!(writer, "{:.17}", bias)?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Load an MLP from a file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        
        let mut lines = content.lines();
        
        // Check format version
        let format = lines.next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing format version")
        })?;
        
        if format != "MLP_SAVE_FORMAT_V1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unsupported file format"
            ));
        }
        
        // Read MLP metadata
        let num_layers: usize = lines.next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing layer count"))?
            .parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid layer count"))?;
        
        // Read execution mode
        let execution_mode_line = lines.next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing execution mode"))?;
        
        let execution_mode = match execution_mode_line {
            "Sequential" => ExecutionMode::Sequential,
            "Parallel" => ExecutionMode::Parallel,
            "SIMD" => ExecutionMode::SIMD,
            "ParallelSIMD" => ExecutionMode::ParallelSIMD,
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid execution mode")),
        };
        
        let learning_rate: f32 = lines.next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing learning rate"))?
            .parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid learning rate"))?;
        
        // Read loss function
        let loss_function = match lines.next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing loss function"))? {
            "MSE" => LossFunctionEnum::MSE,
            "CategoricalCrossEntropy" => LossFunctionEnum::CategoricalCrossEntropy,
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid loss function")),
        };
        
        // Read layers
        let mut layers = Vec::new();
        
        for _ in 0..num_layers {
            // Read layer dimensions
            let dims_line = lines.next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing layer dimensions"))?;
            let dims: Vec<&str> = dims_line.split_whitespace().collect();
            if dims.len() != 2 {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid layer dimensions"));
            }
            
            let rows: usize = dims[0].parse()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid weight rows"))?;
            let cols: usize = dims[1].parse()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid weight cols"))?;
            
            // Read activation type
            let activation = match lines.next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing activation"))? {
                "ReLU" => ActivationType::ReLU,
                "Sigmoid" => ActivationType::Sigmoid,
                "Linear" => ActivationType::Linear,
                "Tanh" => ActivationType::Tanh,
                "Softmax" => ActivationType::Softmax,
                _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid activation")),
            };
            
            // Read weights
            let mut weights_data = Vec::new();
            for _ in 0..(rows * cols) {
                let weight: f32 = lines.next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing weight data"))?
                    .parse()
                    .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid weight value"))?;
                weights_data.push(weight);
            }
            let weights = Tensor::new(weights_data, vec![rows, cols]);
            
            // Read biases
            let mut biases_data = Vec::new();
            for _ in 0..rows {
                let bias: f32 = lines.next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing bias data"))?
                    .parse()
                    .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid bias value"))?;
                biases_data.push(bias);
            }
            let biases = Tensor::new(biases_data, vec![rows, 1]);
            
            // Create layer with proper initialization
            let layer = Layer {
                weights,
                biases,
                activation,
                last_input: None,
                last_pre_activation: None,
                last_output: None,
                loss_function: loss_function.clone()
            };
            
            layers.push(layer);
        }
        
        Ok(MLP {
            layers,
            execution_mode,
            learning_rate,
            loss_function,
        })
    }
}