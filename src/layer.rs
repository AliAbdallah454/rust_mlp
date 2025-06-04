use crate::tensor::Tensor;

pub struct Layer {
    pub weights: Tensor,
}

impl Layer {
    pub fn new(rows: u32, cols: u32, seed: u64) -> Self {
        Self { weights: Tensor::random(rows, cols, seed) }
    }

    pub fn forward(&self, input: &Tensor, nb_threads: usize) -> Tensor {
        self.weights.mul_par(input, nb_threads)
    }
    
}
