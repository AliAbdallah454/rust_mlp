use crate::tensor::Tensor;

impl Tensor {

    pub fn new_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
        Self::new(data, vec![rows, cols])
    }

    pub fn random_2d(rows: usize, cols: usize, seed: u64) -> Self {
        Self::random(vec![rows, cols], seed)
    }

    pub fn ones_2d(rows: usize, cols: usize) -> Tensor {
        Self::ones(vec![rows, cols])
    }

    pub fn zeros_2d(rows: usize, cols: usize) -> Tensor {
        Self::zeros(vec![rows, cols])
    }

}