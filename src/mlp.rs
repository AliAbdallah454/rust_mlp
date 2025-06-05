// use crate::layer::{Layer};
// use crate::tensor::{self, Tensor};

// pub struct MLP {
//     pub layers: Vec<Layer>,
//     pub nb_threads: usize
// }

// impl MLP {
//     pub fn new(layers: Vec<Layer>, nb_threads: usize) -> Self {
//         MLP { 
//             layers,
//             nb_threads
//         }
//     }

//     pub fn forward(&self, input: Tensor) -> Tensor {
//         let mut current_input = input;
//         for layer in &self.layers {
//             current_input = layer.forward(&current_input, self.nb_threads);
//         }
//         current_input
//     }

//     pub fn train_iter(&self, input: Tensor, expect: Tensor) -> (Tensor, Tensor) {
//         let mut current_input = input;
//         for layer in &self.layers {
//             current_input = layer.forward(&current_input, self.nb_threads);
//         }
//         let error = (current_input.clone() - expect).square();
//         (current_input, error)
//     }

//     // Bypass the compiler's mutability check using unsafe code.
//     pub fn inc_weights(&self) {
//         unsafe {
//             let ptr = self.layers.as_ptr() as *mut Layer;
//             let layer1 = &mut *ptr.add(1);
//             layer1.weights.data[0] += 0.1;
//         }
//     }
// }

use crate::layer::{Layer, ActivationType};
use crate::tensor::Tensor;

pub struct MLP {
    pub layers: Vec<Layer>,
    pub nb_threads: usize,
    pub learning_rate: f64,
}

impl MLP {
    pub fn new(layer_sizes: Vec<u32>, activations: Vec<ActivationType>, learning_rate: f64, nb_threads: usize, seed: u64) -> Self {
        assert_eq!(layer_sizes.len() - 1, activations.len(), "Number of activations must match number of layers");
        
        let mut layers = Vec::new();
        let mut current_seed = seed;
        
        for i in 0..layer_sizes.len() - 1 {
            let layer = Layer::new(
                layer_sizes[i], 
                layer_sizes[i + 1], 
                activations[i].clone(),
                current_seed
            );
            layers.push(layer);
            current_seed = current_seed.wrapping_add(1); // Different seed for each layer
        }
        
        MLP {
            layers,
            nb_threads,
            learning_rate,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut current_input = input.clone();
        for layer in &mut self.layers {
            current_input = layer.forward(&current_input, self.nb_threads);
        }
        current_input
    }

    pub fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> f64 {
        // Compute loss
        let loss = prediction.mse_loss(target);
        
        // Compute initial gradient (derivative of loss w.r.t. output)
        let mut gradient = prediction.mse_loss_derivative(target);
        
        // Backpropagate through layers in reverse order
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();
        
        for layer in self.layers.iter().rev() {
            let (dx, dw, db) = layer.backward(&gradient);
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

    pub fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f64 {
        let prediction = self.forward(input);
        self.backward(&prediction, target)
    }

    pub fn train(&mut self, inputs: &[Tensor], targets: &[Tensor], epochs: usize) -> Vec<f64> {
        assert_eq!(inputs.len(), targets.len(), "Number of inputs must match number of targets");
        
        let mut losses = Vec::new();
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let loss = self.train_step(input, target);
                epoch_loss += loss;
            }
            
            epoch_loss /= inputs.len() as f64;
            losses.push(epoch_loss);
            
            // if epoch % 5 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, epoch_loss);
            // }
        }
        
        losses
    }

    pub fn predict(&mut self, input: &Tensor) -> Tensor {
        self.forward(input)
    }
}