use std::ops::{Add, Sub};

use crate::tensor::Tensor;

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Tensor add: shape mismatch {:?} vs {:?}", self.shape, rhs.shape);
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a + b).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Tensor sub: shape mismatch {:?} vs {:?}", self.shape, rhs.shape);
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Tensor sub: shape mismatch {:?} vs {:?}", self.shape, rhs.shape);
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let epsilon = 1e-2;
        if self.shape != other.shape {
            return false;
        }

        self.data.iter()
            .zip(&other.data)
            .all(|(a, b)| (a - b).abs() < epsilon)
    }
}