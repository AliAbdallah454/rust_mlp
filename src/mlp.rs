use crate::layer::{Layer};
use crate::tensor::Tensor;

pub struct MLP {
    pub layers: Vec<Layer>,
    pub nb_threads: usize
}

impl MLP {
    pub fn new(layers: Vec<Layer>, nb_threads: usize) -> Self {
        MLP { 
            layers,
            nb_threads
        }
    }

    pub fn forward(&self, input: Tensor) -> Tensor {
        let mut current_input = input;
        for layer in &self.layers {
            current_input = layer.forward(&current_input, self.nb_threads);
        }
        current_input
    }

}