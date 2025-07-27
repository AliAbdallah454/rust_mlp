use rust_mlp::helpers::split_dataset;
use rust_mlp::layer::ActivationType;
use rust_mlp::mlp::{LossFunction, MLP};
use rust_mlp::mnits_data::MnistData;
use rust_mlp::tensor::{ExecutionMode, Tensor};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc::{channel, Receiver, Sender};

// GUI dependencies
use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};

#[derive(Debug, Clone)]
struct TrainingProgress {
    current_epoch: usize,
    total_epochs: usize,
    accuracy: f32,
    is_complete: bool,
    execution_mode: ExecutionMode,
    training_time: Option<std::time::Duration>,
}

struct MnistApp {
    mlp: Arc<Mutex<Option<MLP>>>,
    training_complete: bool,
    canvas_image: Vec<f32>,
    canvas_size: usize,
    is_drawing: bool,
    last_pos: Option<Pos2>,
    predicted_digit: Option<usize>,
    prediction_confidence: Vec<f32>,
    training_status: String,
    
    // New fields for enhanced functionality
    total_epochs: usize,
    current_epoch: usize,
    selected_execution_mode: ExecutionMode,
    training_progress_receiver: Option<Receiver<TrainingProgress>>,
    training_accuracies: Vec<f32>,
    execution_mode_results: Vec<(ExecutionMode, std::time::Duration, f32)>,
    show_mode_comparison: bool,
}

impl Default for MnistApp {
    fn default() -> Self {
        Self {
            mlp: Arc::new(Mutex::new(None)),
            training_complete: false,
            canvas_image: vec![0.0; 28 * 28],
            canvas_size: 28,
            is_drawing: false,
            last_pos: None,
            predicted_digit: None,
            prediction_confidence: vec![0.0; 10],
            training_status: "Select execution mode and click 'Start Training' to begin".to_string(),
            
            // Initialize new fields
            total_epochs: 10,
            current_epoch: 0,
            selected_execution_mode: ExecutionMode::Sequential,
            training_progress_receiver: None,
            training_accuracies: Vec::new(),
            execution_mode_results: Vec::new(),
            show_mode_comparison: false,
        }
    }
}

impl eframe::App for MnistApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for training progress updates
        if let Some(receiver) = &self.training_progress_receiver {
            if let Ok(progress) = receiver.try_recv() {
                self.current_epoch = progress.current_epoch;
                self.training_status = if progress.is_complete {
                    format!("Training complete! Final accuracy: {:.2}% (Time: {:.2?})", 
                           progress.accuracy, progress.training_time.unwrap_or_default())
                } else {
                    format!("Training... Epoch {}/{} - Accuracy: {:.2}%", 
                           progress.current_epoch, progress.total_epochs, progress.accuracy)
                };
                
                if progress.accuracy > 0.0 {
                    self.training_accuracies.push(progress.accuracy);
                }
                
                if progress.is_complete {
                    self.training_complete = true;
                    self.training_progress_receiver = None;
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("MNIST Digit Recognition with Neural Network");
            ui.label("Architecture: 784 → 16 → 16 → 10 (Compact Model)");
            
            ui.separator();
            
            // Configuration section
            ui.horizontal(|ui| {
                ui.label("Epochs:");
                ui.add(egui::Slider::new(&mut self.total_epochs, 1..=50));
            });
            
            ui.horizontal(|ui| {
                ui.label("Execution Mode:");
                egui::ComboBox::from_label("")
                    .selected_text(format!("{:?}", self.selected_execution_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_execution_mode, ExecutionMode::Sequential, "Sequential");
                        ui.selectable_value(&mut self.selected_execution_mode, ExecutionMode::Parallel, "Parallel");
                        ui.selectable_value(&mut self.selected_execution_mode, ExecutionMode::SIMD, "SIMD");
                        ui.selectable_value(&mut self.selected_execution_mode, ExecutionMode::ParallelSIMD, "ParallelSIMD");
                    });
            });
            
            ui.separator();
            
            // Training section
            ui.horizontal(|ui| {
                let training_in_progress = self.training_progress_receiver.is_some();
                
                if ui.add_enabled(!training_in_progress, egui::Button::new("Start Training")).clicked() {
                    self.start_training();
                }
                
                if ui.add_enabled(!training_in_progress, egui::Button::new("Compare All Modes")).clicked() {
                    self.start_mode_comparison();
                }
                
                ui.label(&self.training_status);
            });
            
            // Training progress bar
            if self.training_progress_receiver.is_some() {
                ui.horizontal(|ui| {
                    ui.label("Progress:");
                    let progress = if self.total_epochs > 0 {
                        self.current_epoch as f32 / self.total_epochs as f32
                    } else {
                        0.0
                    };
                    ui.add(egui::ProgressBar::new(progress)
                           .text(format!("{}/{}", self.current_epoch, self.total_epochs)));
                });
            }
            
            // Accuracy plot
            if !self.training_accuracies.is_empty() {
                ui.separator();
                ui.label("Training Accuracy Over Time:");
                
                let plot_height = 100.0;
                let (response, painter) = ui.allocate_painter(
                    Vec2::new(ui.available_width(), plot_height),
                    egui::Sense::hover()
                );
                
                self.draw_accuracy_plot(&painter, response.rect);
            }
            
            // Mode comparison results
            if self.show_mode_comparison && !self.execution_mode_results.is_empty() {
                ui.separator();
                ui.heading("Execution Mode Comparison:");
                
                let sequential_time = self.execution_mode_results.iter()
                    .find(|(mode, _, _)| matches!(mode, ExecutionMode::Sequential))
                    .map(|(_, time, _)| time.as_secs_f64())
                    .unwrap_or(1.0);
                
                for (mode, duration, accuracy) in &self.execution_mode_results {
                    let speedup = sequential_time / duration.as_secs_f64();
                    ui.horizontal(|ui| {
                        ui.label(format!("{:?}:", mode));
                        ui.label(format!("Time: {:.2?}", duration));
                        ui.label(format!("Speedup: {:.2}x", speedup));
                        ui.label(format!("Accuracy: {:.2}%", accuracy));
                    });
                }
            }
            
            ui.separator();
            
            // Drawing section - only enabled when training is complete
            if self.training_complete {
                ui.heading("Draw a digit (0-9):");
                
                // Drawing canvas
                let canvas_size = 280.0; // 10x scale for visibility
                let (response, painter) = ui.allocate_painter(
                    Vec2::splat(canvas_size),
                    egui::Sense::drag()
                );
                
                // Draw canvas background
                painter.rect_filled(
                    response.rect,
                    0.0,
                    Color32::BLACK
                );
                
                // Handle drawing
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.draw_on_canvas(pos, response.rect, canvas_size);
                        self.is_drawing = true;
                    }
                } else {
                    self.is_drawing = false;
                    self.last_pos = None;
                }
                
                // Draw the current canvas state
                self.draw_canvas(&painter, response.rect, canvas_size);
                
                ui.horizontal(|ui| {
                    if ui.button("Clear Canvas").clicked() {
                        self.clear_canvas();
                    }
                    
                    if ui.button("Predict Digit").clicked() {
                        self.predict_digit();
                    }
                });
                
                // Display prediction results
                if let Some(digit) = self.predicted_digit {
                    ui.separator();
                    ui.heading(format!("Predicted Digit: {}", digit));
                    
                    ui.label("Confidence scores:");
                    for (i, &confidence) in self.prediction_confidence.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("Digit {}: ", i));
                            ui.add(egui::ProgressBar::new(confidence).text(format!("{:.1}%", confidence * 100.0)));
                        });
                    }
                }
            } else if self.training_progress_receiver.is_none() {
                ui.label("Complete training to enable digit drawing and prediction.");
            } else {
                ui.label("Training in progress... Please wait.");
            }
        });
        
        // Request repaint for smooth drawing and progress updates
        if self.is_drawing || self.training_progress_receiver.is_some() {
            ctx.request_repaint();
        }
    }
}

impl MnistApp {
    fn start_training(&mut self) {
        let (sender, receiver) = channel::<TrainingProgress>();
        self.training_progress_receiver = Some(receiver);
        self.current_epoch = 0;
        self.training_complete = false;
        self.training_accuracies.clear();
        self.training_status = "Loading MNIST data...".to_string();
        
        let mlp_arc = Arc::clone(&self.mlp);
        let total_epochs = self.total_epochs;
        let execution_mode = self.selected_execution_mode;
        
        thread::spawn(move || {
            Self::train_model(mlp_arc, total_epochs, execution_mode, sender);
        });
    }
    
    fn start_mode_comparison(&mut self) {
        self.execution_mode_results.clear();
        self.show_mode_comparison = true;
        self.training_status = "Comparing execution modes...".to_string();
        
        let (sender, receiver) = channel::<TrainingProgress>();
        self.training_progress_receiver = Some(receiver);
        
        let mlp_arc = Arc::clone(&self.mlp);
        let total_epochs = 5; // Use fewer epochs for comparison
        
        thread::spawn(move || {
            let execution_modes = vec![
                ExecutionMode::Sequential,
                ExecutionMode::Parallel,
                ExecutionMode::SIMD,
                ExecutionMode::ParallelSIMD,
            ];
            
            let mut results = Vec::new();
            
            for (i, &mode) in execution_modes.iter().enumerate() {
                sender.send(TrainingProgress {
                    current_epoch: i + 1,
                    total_epochs: execution_modes.len(),
                    accuracy: 0.0,
                    is_complete: false,
                    execution_mode: mode,
                    training_time: None,
                }).ok();
                
                let (duration, accuracy) = Self::train_single_mode(mode, total_epochs);
                results.push((mode, duration, accuracy));
            }
            
            // Send completion signal
            sender.send(TrainingProgress {
                current_epoch: execution_modes.len(),
                total_epochs: execution_modes.len(),
                accuracy: results.last().map(|(_, _, acc)| *acc).unwrap_or(0.0),
                is_complete: true,
                execution_mode: ExecutionMode::Sequential,
                training_time: Some(results.iter().map(|(_, dur, _)| *dur).sum()),
            }).ok();
        });
    }
    
    fn train_single_mode(execution_mode: ExecutionMode, epochs: usize) -> (std::time::Duration, f32) {
        // Load and prepare data (simplified for comparison)
        if let Ok(data) = MnistData::load_from_files(
            "./mnist/train-images.idx3-ubyte",
            "./mnist/train-labels.idx1-ubyte"
        ) {
            let mut images: Vec<Tensor> = Vec::with_capacity(1000); // Use smaller dataset
            let mut labels: Vec<Tensor> = Vec::with_capacity(1000);
            
            for (i, image) in data.images.iter().take(1000).enumerate() {
                let normalized: Vec<f32> = image.iter()
                    .map(|&pixel| pixel as f32 / 255.0)
                    .collect();
                images.push(Tensor::new(normalized, 28 * 28, 1));
                
                let mut one_hot = vec![0.0; 10];
                one_hot[data.labels[i] as usize] = 1.0;
                labels.push(Tensor::new(one_hot, 10, 1));
            }
            
            let (training_images, training_labels, testing_images, testing_labels) = 
                split_dataset(images, labels, 0.8);
            
            // Changed architecture: 784 -> 16 -> 16 -> 10
            let layer_sizes = vec![28 * 28, 16, 16, 10];
            let activations = vec![
                ActivationType::ReLU, 
                ActivationType::ReLU, 
                ActivationType::Softmax
            ];
            
            let mut mlp = MLP::new(
                layer_sizes,
                activations,
                LossFunction::CategoricalCrossEntropy,
                0.01,
                execution_mode,
                42
            );
            
            let start_time = Instant::now();
            
            for _ in 0..epochs {
                mlp.train(&training_images, &training_labels, 1);
            }
            
            let duration = start_time.elapsed();
            
            // Calculate final accuracy
            let mut correct = 0;
            for (test_img, test_label) in testing_images.iter().zip(testing_labels.iter()) {
                let prediction = mlp.forward(test_img);
                let predicted_class = prediction.data.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap().0;
                
                let actual_class = test_label.data.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap().0;
                
                if predicted_class == actual_class {
                    correct += 1;
                }
            }
            
            let accuracy = correct as f32 / testing_images.len() as f32 * 100.0;
            (duration, accuracy)
        } else {
            (std::time::Duration::from_secs(0), 0.0)
        }
    }
    
    fn train_model(
        mlp_arc: Arc<Mutex<Option<MLP>>>, 
        total_epochs: usize, 
        execution_mode: ExecutionMode,
        sender: Sender<TrainingProgress>
    ) {
        // Load MNIST data
        let train_data_result = MnistData::load_from_files(
            "./mnist/train-images.idx3-ubyte",
            "./mnist/train-labels.idx1-ubyte"
        );
        
        match train_data_result {
            Ok(data) => {
                // Convert images and labels to tensors
                let mut images: Vec<Tensor> = Vec::with_capacity(data.images.len());
                let mut labels: Vec<Tensor> = Vec::with_capacity(data.labels.len());
                
                for image in data.images.iter() {
                    // Normalize pixel values to [0, 1]
                    let normalized: Vec<f32> = image.iter()
                        .map(|&pixel| pixel as f32 / 255.0)
                        .collect();
                    images.push(Tensor::new(normalized, 28 * 28, 1));
                }
                
                for label in data.labels.iter() {
                    let mut one_hot = vec![0.0; 10];
                    one_hot[*label as usize] = 1.0;
                    labels.push(Tensor::new(one_hot, 10, 1));
                }
                
                // Use a subset for faster training
                let subset_size = std::cmp::min(10000, images.len());
                images.truncate(subset_size);
                labels.truncate(subset_size);
                
                let (training_images, training_labels, testing_images, testing_labels) = 
                    split_dataset(images, labels, 0.8);
                
                // Changed architecture: 784 -> 16 -> 16 -> 10
                let layer_sizes = vec![28 * 28, 16, 16, 10];
                let activations = vec![
                    ActivationType::ReLU, 
                    ActivationType::ReLU, 
                    ActivationType::Softmax
                ];
                
                let mut mlp = MLP::new(
                    layer_sizes,
                    activations,
                    LossFunction::CategoricalCrossEntropy,
                    0.01,
                    execution_mode,
                    42
                );
                
                // Train the model
                let train_start = Instant::now();
                
                for epoch in 0..total_epochs {
                    mlp.train(&training_images, &training_labels, 1);
                    
                    // Calculate accuracy on test set
                    let mut correct = 0;
                    for (test_img, test_label) in testing_images.iter().zip(testing_labels.iter()) {
                        let prediction = mlp.forward(test_img);
                        let predicted_class = prediction.data.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .unwrap().0;
                        
                        let actual_class = test_label.data.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .unwrap().0;
                        
                        if predicted_class == actual_class {
                            correct += 1;
                        }
                    }
                    
                    let accuracy = correct as f32 / testing_images.len() as f32 * 100.0;
                    
                    // Send progress update
                    sender.send(TrainingProgress {
                        current_epoch: epoch + 1,
                        total_epochs,
                        accuracy,
                        is_complete: false,
                        execution_mode,
                        training_time: None,
                    }).ok();
                }
                
                let duration = train_start.elapsed();
                
                // Store the trained model
                *mlp_arc.lock().unwrap() = Some(mlp);
                
                // Send completion signal
                sender.send(TrainingProgress {
                    current_epoch: total_epochs,
                    total_epochs,
                    accuracy: 0.0, // Final accuracy will be calculated separately
                    is_complete: true,
                    execution_mode,
                    training_time: Some(duration),
                }).ok();
            }
            Err(e) => {
                println!("Error loading MNIST data: {:?}", e);
            }
        }
    }
    
    fn draw_accuracy_plot(&self, painter: &egui::Painter, rect: Rect) {
        if self.training_accuracies.is_empty() {
            return;
        }
        
        let max_accuracy = self.training_accuracies.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_accuracy = self.training_accuracies.iter().fold(100.0f32, |a, &b| a.min(b));
        let accuracy_range = (max_accuracy - min_accuracy).max(1.0);
        
        // Draw axes
        painter.line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            Stroke::new(1.0, Color32::WHITE)
        );
        painter.line_segment(
            [rect.left_bottom(), rect.left_top()],
            Stroke::new(1.0, Color32::WHITE)
        );
        
        // Draw accuracy line
        if self.training_accuracies.len() > 1 {
            let points: Vec<Pos2> = self.training_accuracies.iter().enumerate()
                .map(|(i, &acc)| {
                    let x = rect.left() + (i as f32 / (self.training_accuracies.len() - 1) as f32) * rect.width();
                    let y = rect.bottom() - ((acc - min_accuracy) / accuracy_range) * rect.height();
                    Pos2::new(x, y)
                })
                .collect();
            
            for window in points.windows(2) {
                painter.line_segment(
                    [window[0], window[1]],
                    Stroke::new(2.0, Color32::GREEN)
                );
            }
        }
    }
    
    fn draw_on_canvas(&mut self, pos: Pos2, canvas_rect: Rect, canvas_size: f32) {
        let relative_pos = pos - canvas_rect.min;
        let normalized_x = (relative_pos.x / canvas_size).clamp(0.0, 1.0);
        let normalized_y = (relative_pos.y / canvas_size).clamp(0.0, 1.0);
        
        let pixel_x = (normalized_x * 28.0) as usize;
        let pixel_y = (normalized_y * 28.0) as usize;
        
        if pixel_x < 28 && pixel_y < 28 {
            // Draw with a brush effect (set multiple pixels)
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = pixel_x as i32 + dx;
                    let ny = pixel_y as i32 + dy;
                    
                    if nx >= 0 && nx < 28 && ny >= 0 && ny < 28 {
                        let idx = (ny as usize) * 28 + (nx as usize);
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        let intensity = (1.0 - distance / 2.0).max(0.0);
                        self.canvas_image[idx] = (self.canvas_image[idx] + intensity * 0.8).min(1.0);
                    }
                }
            }
        }
    }
    
    fn draw_canvas(&self, painter: &egui::Painter, canvas_rect: Rect, canvas_size: f32) {
        let pixel_size = canvas_size / 28.0;
        
        for y in 0..28 {
            for x in 0..28 {
                let intensity = self.canvas_image[y * 28 + x];
                if intensity > 0.0 {
                    let color = Color32::from_gray((intensity * 255.0) as u8);
                    let pixel_rect = Rect::from_min_size(
                        canvas_rect.min + Vec2::new(x as f32 * pixel_size, y as f32 * pixel_size),
                        Vec2::splat(pixel_size)
                    );
                    painter.rect_filled(pixel_rect, 0.0, color);
                }
            }
        }
    }
    
    fn clear_canvas(&mut self) {
        self.canvas_image.fill(0.0);
        self.predicted_digit = None;
        self.prediction_confidence.fill(0.0);
    }
    
    fn predict_digit(&mut self) {
        if let Some(mlp) = self.mlp.lock().unwrap().as_mut(){
            // Create tensor from canvas image
            let input_tensor = Tensor::new(self.canvas_image.clone(), 28 * 28, 1);
            
            // Get prediction
            let output = mlp.forward(&input_tensor);
            
            // Find the predicted digit (highest probability)
            let predicted_class = output.data.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0;
            
            self.predicted_digit = Some(predicted_class);
            self.prediction_confidence = output.data.clone();
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([700.0, 900.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Enhanced MNIST Neural Network Trainer",
        options,
        Box::new(|_cc| Ok(Box::new(MnistApp::default())))
    )
}