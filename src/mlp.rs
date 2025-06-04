use crate::layer::{Layer};
use crate::tensor::{self, Tensor};

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

    pub fn train_iter(&self, input: Tensor, expect: Tensor) -> (Tensor, Tensor) {
        let mut current_input = input;
        for layer in &self.layers {
            current_input = layer.forward(&current_input, self.nb_threads);
        }
        let error = (current_input.clone() - expect).square();
        (current_input, error)
    }

    // Bypass the compiler's mutability check using unsafe code.
    pub fn inc_weights(&self) {
        unsafe {
            let ptr = self.layers.as_ptr() as *mut Layer;
            let layer1 = &mut *ptr.add(1);
            layer1.weights.data[0] += 0.1;
        }
    }
}