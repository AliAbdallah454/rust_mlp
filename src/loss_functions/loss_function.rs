use crate::tensor::Tensor;

pub trait LossFunction {

    fn forward(predicted: &Tensor, target: &Tensor) -> f32;
    fn backward(predicted: &Tensor, target: &Tensor) -> Tensor;

}

pub struct MSE;

impl LossFunction for MSE {

    fn forward(predicted: &Tensor, target: &Tensor) -> f32 {
        let diff = predicted - target;
        let squared = diff.square();
        squared.sum() / predicted.size() as f32
    }

    fn backward(predicted: &Tensor, target: &Tensor) -> Tensor {
        let diff = predicted - target;
        diff.scale(2.0 / predicted.size() as f32)
    }

}

pub struct CategoricalCrossEntropy;

impl LossFunction for CategoricalCrossEntropy  {
    
    fn forward(predicted: &Tensor, target: &Tensor) -> f32 {
        assert_eq!(predicted.shape, target.shape, "Shape mismatch {:?} vs {:?}", predicted.shape, target.shape);
    
        let epsilon = 1e-15; // To prevent log(0)
        let mut total_loss = 0.0;
    
        for i in 0..predicted.size() {
            let y_true = target.data[i];
            let y_pred = predicted.data[i].max(epsilon).min(1.0 - epsilon); // clip
            total_loss -= y_true * y_pred.ln();
        }
    
        total_loss / predicted.rows() as f32 // Can be changed ...
    }

    fn backward(predicted: &Tensor, target: &Tensor) -> Tensor {
        assert_eq!(predicted.shape, target.shape, "Shape mismatch {:?} vs {:?}", predicted.shape, target.shape);
        assert_eq!(predicted.cols(), 1, "Categorical cross entropy derivative only implemented for column vectors");

        // predicted: softmax output, target: one-hot labels
        let diff = predicted - target;
        diff.scale(1.0 / predicted.rows() as f32) // Can be changed ...
    }

}