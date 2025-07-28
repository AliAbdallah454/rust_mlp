use crate::tensor::Tensor;

pub trait ActivationFunction {

    fn activate(&self, layer: &Tensor) -> Tensor;
    fn derivative(&self, layer: &Tensor) -> Tensor;

}