use crate::tensor::Tensor;
use crate::mlp::MLP;

use rand::seq::SliceRandom;
use rand::SeedableRng;

pub fn evaluate_model(mlp: &mut MLP, images: &[Tensor], labels: &[Tensor]) -> f64 {
    let mut correct = 0;
    for (img, label) in images.iter().zip(labels.iter()) {
        let prediction = mlp.predict(img);
        let predicted_class = prediction.argmax();
        let true_class = label.argmax();

        if predicted_class == true_class {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / images.len() as f64;
    accuracy
}
pub fn split_dataset(
    images: Vec<Tensor>,
    labels: Vec<Tensor>,
    train_ratio: f64,
) -> (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>) {
    assert_eq!(images.len(), labels.len(), "Images and labels length mismatch");

    let mut indices: Vec<usize> = (0..images.len()).collect();
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    indices.shuffle(&mut rng);

    let train_size = (train_ratio * images.len() as f64).round() as usize;

    let mut train_images = Vec::with_capacity(train_size);
    let mut train_labels = Vec::with_capacity(train_size);
    let mut test_images = Vec::with_capacity(images.len() - train_size);
    let mut test_labels = Vec::with_capacity(images.len() - train_size);

    for (i, &idx) in indices.iter().enumerate() {
        if i < train_size {
            train_images.push(images[idx].clone());
            train_labels.push(labels[idx].clone());
        } else {
            test_images.push(images[idx].clone());
            test_labels.push(labels[idx].clone());
        }
    }

    (train_images, train_labels, test_images, test_labels)
}