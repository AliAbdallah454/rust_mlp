use cp_proj::helpers::split_dataset;
use cp_proj::layer::ActivationType;
use cp_proj::mlp::{LossFunction, MLP};
use cp_proj::mnits_data::MnistData;
use cp_proj::tensor::{ExecutionMode, Tensor};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc::{channel, Receiver, Sender};

// GUI dependencies
use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};

#[derive(Debug, Clone, PartialEq)]
pub enum TrainingMode {
    FullDataset,
    MiniBatch,
    Stochastic,
}

impl TrainingMode {
    fn as_str(&self) -> &'static str {
        match self {
            TrainingMode::FullDataset => "Full Dataset",
            TrainingMode::MiniBatch => "Mini-Batch",
            TrainingMode::Stochastic => "Stochastic",
        }
    }
}

#[derive(Debug, Clone)]
struct TrainingProgress {
    current_epoch: usize,
    total_epochs: usize,
    accuracy: f32,
    is_complete: bool,
    execution_mode: ExecutionMode,
    training_time: Option<std::time::Duration>,
}

#[derive(Debug, Clone)]
struct LayerConfig {
    size: usize,
    activation: ActivationType,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            size: 16,
            activation: ActivationType::ReLU,
        }
    }
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
    
    // Enhanced configuration fields
    total_epochs: usize,
    current_epoch: usize,
    selected_execution_mode: ExecutionMode,
    selected_training_mode: TrainingMode,
    selected_loss_function: LossFunction,
    learning_rate: f32,
    batch_size: usize,
    training_progress_receiver: Option<Receiver<TrainingProgress>>,
    training_accuracies: Vec<f32>,
    execution_mode_results: Vec<(ExecutionMode, std::time::Duration, f32)>,
    show_mode_comparison: bool,
    
    // Model management
    model_save_path: String,
    model_load_path: String,
    show_save_dialog: bool,
    show_load_dialog: bool,
    
    // Architecture configuration
    hidden_layers: Vec<LayerConfig>,
    show_architecture_config: bool,
    input_size: usize,
    output_size: usize,
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
            training_status: "Configure architecture and training parameters, then click 'Start Training'".to_string(),
            
            // Enhanced configuration
            total_epochs: 10,
            current_epoch: 0,
            selected_execution_mode: ExecutionMode::Sequential,
            selected_training_mode: TrainingMode::MiniBatch,
            selected_loss_function: LossFunction::CategoricalCrossEntropy,
            learning_rate: 0.01,
            batch_size: 32,
            training_progress_receiver: None,
            training_accuracies: Vec::new(),
            execution_mode_results: Vec::new(),
            show_mode_comparison: false,
            
            // Model management
            model_save_path: "mnist_model.txt".to_string(),
            model_load_path: "mnist_model.txt".to_string(),
            show_save_dialog: false,
            show_load_dialog: false,
            
            // Architecture configuration
            hidden_layers: vec![
                LayerConfig { size: 128, activation: ActivationType::ReLU },
                LayerConfig { size: 64, activation: ActivationType::ReLU },
            ],
            show_architecture_config: false,
            input_size: 784,
            output_size: 10,
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
            ui.heading("Enhanced MNIST Neural Network Trainer");
            
            // Model Management Section
            ui.separator();
            ui.heading("Model Management");
            ui.horizontal(|ui| {
                if ui.button("💾 Save Model").clicked() {
                    self.show_save_dialog = true;
                }
                if ui.button("📂 Load Model").clicked() {
                    self.show_load_dialog = true;
                }
                if ui.button("🏗️ Configure Architecture").clicked() {
                    self.show_architecture_config = true;
                }
            });
            
            // Architecture Display
            ui.horizontal(|ui| {
                ui.label("Current Architecture:");
                ui.label(format!("{}", self.input_size));
                for layer in &self.hidden_layers {
                    ui.label("→");
                    ui.label(format!("{} ({:?})", layer.size, layer.activation));
                }
                ui.label("→");
                ui.label(format!("{} (Softmax)", self.output_size));
            });
            
            ui.separator();
            
            // Training Configuration Section
            ui.heading("Training Configuration");
            
            ui.horizontal(|ui| {
                ui.label("Epochs:");
                ui.add(egui::Slider::new(&mut self.total_epochs, 1..=100));
            });
            
            ui.horizontal(|ui| {
                ui.label("Learning Rate:");
                ui.add(egui::Slider::new(&mut self.learning_rate, 0.001..=0.1)
                       .logarithmic(true)
                       .step_by(0.001));
            });
            
            ui.horizontal(|ui| {
                ui.label("Training Mode:");
                egui::ComboBox::from_id_salt("training_mode_combo")  // Added unique ID
                    .selected_text(self.selected_training_mode.as_str())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_training_mode, TrainingMode::FullDataset, "Full Dataset");
                        ui.selectable_value(&mut self.selected_training_mode, TrainingMode::MiniBatch, "Mini-Batch");
                        ui.selectable_value(&mut self.selected_training_mode, TrainingMode::Stochastic, "Stochastic");
                    });
            });
            
            if self.selected_training_mode == TrainingMode::MiniBatch {
                ui.horizontal(|ui| {
                    ui.label("Batch Size:");
                    ui.add(egui::Slider::new(&mut self.batch_size, 1..=256));
                });
            }
            
            ui.horizontal(|ui| {
                ui.label("Execution Mode:");
                egui::ComboBox::from_id_salt("execution_mode_combo")  // Added unique ID
                    .selected_text(format!("{:?}", self.selected_execution_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_execution_mode, ExecutionMode::Sequential, "Sequential");
                        ui.selectable_value(&mut self.selected_execution_mode, ExecutionMode::Parallel, "Parallel");
                        ui.selectable_value(&mut self.selected_execution_mode, ExecutionMode::SIMD, "SIMD");
                        ui.selectable_value(&mut self.selected_execution_mode, ExecutionMode::ParallelSIMD, "ParallelSIMD");
                    });
            });
            
            ui.horizontal(|ui| {
                ui.label("Loss Function:");
                egui::ComboBox::from_id_salt("loss_function_combo")  // Added unique ID
                    .selected_text(format!("{:?}", self.selected_loss_function))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_loss_function, LossFunction::CategoricalCrossEntropy, "Categorical Cross Entropy");
                        ui.selectable_value(&mut self.selected_loss_function, LossFunction::MSE, "Mean Squared Error");
                    });
            });
            
            
            ui.separator();
            
            // Training Control Section
            ui.horizontal(|ui| {
                let training_in_progress = self.training_progress_receiver.is_some();
                
                if ui.add_enabled(!training_in_progress, egui::Button::new("🚀 Start Training")).clicked() {
                    self.start_training();
                }
                
                if ui.add_enabled(!training_in_progress, egui::Button::new("📊 Compare All Modes")).clicked() {
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
                    if ui.button("🗑️ Clear Canvas").clicked() {
                        self.clear_canvas();
                    }
                    
                    if ui.button("🔍 Predict Digit").clicked() {
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
        
        // Show dialogs
        self.show_save_model_dialog(ctx);
        self.show_load_model_dialog(ctx);
        self.show_architecture_config_dialog(ctx);
        
        // Request repaint for smooth drawing and progress updates
        if self.is_drawing || self.training_progress_receiver.is_some() {
            ctx.request_repaint();
        }
    }
}

impl MnistApp {
    fn show_save_model_dialog(&mut self, ctx: &egui::Context) {
        if self.show_save_dialog {
            egui::Window::new("💾 Save Model")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Enter filename to save the trained model:");
                    ui.text_edit_singleline(&mut self.model_save_path);
                    
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            self.save_model();
                            self.show_save_dialog = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_save_dialog = false;
                        }
                    });
                });
        }
    }
    
    fn show_load_model_dialog(&mut self, ctx: &egui::Context) {
        if self.show_load_dialog {
            egui::Window::new("📂 Load Model")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Enter filename to load a trained model:");
                    ui.text_edit_singleline(&mut self.model_load_path);
                    
                    ui.horizontal(|ui| {
                        if ui.button("Load").clicked() {
                            self.load_model();
                            self.show_load_dialog = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_load_dialog = false;
                        }
                    });
                });
        }
    }
    
    fn show_architecture_config_dialog(&mut self, ctx: &egui::Context) {
        if self.show_architecture_config {
            egui::Window::new("🏗️ Configure Architecture")
                .collapsible(false)
                .resizable(true)
                .default_width(400.0)
                .show(ctx, |ui| {
                    ui.label("Neural Network Architecture Configuration");
                    ui.separator();
                    
                    // Input layer (fixed for MNIST)
                    ui.horizontal(|ui| {
                        ui.label("Input Layer:");
                        ui.label(format!("{} neurons", self.input_size));
                    });
                    
                    // Hidden layers configuration
                    ui.label("Hidden Layers:");
                    
                    let mut layers_to_remove = Vec::new();
                    for (i, layer) in self.hidden_layers.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("Layer {}:", i + 1));
                            ui.add(egui::Slider::new(&mut layer.size, 1..=1024));
                            
                            egui::ComboBox::from_id_salt(format!("activation_{}", i))
                                .selected_text(format!("{:?}", layer.activation))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut layer.activation, ActivationType::ReLU, "ReLU");
                                    ui.selectable_value(&mut layer.activation, ActivationType::Sigmoid, "Sigmoid");
                                    ui.selectable_value(&mut layer.activation, ActivationType::Tanh, "Tanh");
                                    ui.selectable_value(&mut layer.activation, ActivationType::Linear, "Linear");
                                });
                            
                            if ui.button("❌").clicked() {
                                layers_to_remove.push(i);
                            }
                        });
                    }
                    
                    // Remove layers marked for deletion
                    for &i in layers_to_remove.iter().rev() {
                        self.hidden_layers.remove(i);
                    }
                    
                    // Add layer button
                    if ui.button("➕ Add Hidden Layer").clicked() {
                        self.hidden_layers.push(LayerConfig::default());
                    }
                    
                    ui.separator();
                    
                    // Output layer (fixed for MNIST)
                    ui.horizontal(|ui| {
                        ui.label("Output Layer:");
                        ui.label(format!("{} neurons (Softmax)", self.output_size));
                    });
                    
                    ui.separator();
                    
                    // Action buttons
                    ui.horizontal(|ui| {
                        if ui.button("Apply Configuration").clicked() {
                            // Reset training state when architecture changes
                            self.training_complete = false;
                            *self.mlp.lock().unwrap() = None;
                            self.training_accuracies.clear();
                            self.training_status = "Architecture updated. Start training with new configuration.".to_string();
                            self.show_architecture_config = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_architecture_config = false;
                        }
                    });
                });
        }
    }
    
    fn save_model(&mut self) {
        if let Some(ref mlp) = *self.mlp.lock().unwrap() {
            match mlp.save(&self.model_save_path) {
                Ok(()) => {
                    self.training_status = format!("Model saved successfully to: {}", self.model_save_path);
                }
                Err(e) => {
                    self.training_status = format!("Error saving model: {}", e);
                }
            }
        } else {
            self.training_status = "No trained model to save. Train a model first.".to_string();
        }
    }
    
    fn load_model(&mut self) {
        match MLP::load(&self.model_load_path) {
            Ok(loaded_mlp) => {
                // Update GUI state based on loaded model
                self.learning_rate = loaded_mlp.learning_rate;
                self.selected_execution_mode = loaded_mlp.execution_mode;
                self.selected_loss_function = loaded_mlp.loss_function.clone();
                
                // Update architecture display based on loaded model
                self.hidden_layers.clear();
                for layer in &loaded_mlp.layers[..loaded_mlp.layers.len()-1] { // Exclude output layer
                    self.hidden_layers.push(LayerConfig {
                        size: layer.weights.rows,
                        activation: layer.activation.clone(),
                    });
                }
                
                *self.mlp.lock().unwrap() = Some(loaded_mlp);
                self.training_complete = true;
                self.training_status = format!("Model loaded successfully from: {}", self.model_load_path);
            }
            Err(e) => {
                self.training_status = format!("Error loading model: {}", e);
            }
        }
    }
    
    fn get_layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![self.input_size];
        for layer in &self.hidden_layers {
            sizes.push(layer.size);
        }
        sizes.push(self.output_size);
        sizes
    }
    
    fn get_activations(&self) -> Vec<ActivationType> {
        let mut activations = Vec::new();
        for layer in &self.hidden_layers {
            activations.push(layer.activation.clone());
        }
        activations.push(ActivationType::Softmax); // Output layer always uses Softmax for classification
        activations
    }
    
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
        let training_mode = self.selected_training_mode.clone();
        let learning_rate = self.learning_rate;
        let loss_function = self.selected_loss_function.clone();
        let batch_size = self.batch_size;
        let layer_sizes = self.get_layer_sizes();
        let activations = self.get_activations();
        
        thread::spawn(move || {
            Self::train_model(
                mlp_arc, 
                total_epochs, 
                execution_mode, 
                training_mode,
                learning_rate,
                loss_function,
                batch_size,
                layer_sizes,
                activations,
                sender
            );
        });
    }
    
    fn start_mode_comparison(&mut self) {
        self.execution_mode_results.clear();
        self.show_mode_comparison = true;
        self.training_status = "Comparing execution modes...".to_string();
        
        let (sender, receiver) = channel::<TrainingProgress>();
        self.training_progress_receiver = Some(receiver);
        
        let layer_sizes = self.get_layer_sizes();
        let activations = self.get_activations();
        let learning_rate = self.learning_rate;
        let loss_function = self.selected_loss_function.clone();
        
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
                
                let (duration, accuracy) = Self::train_single_mode(
                    mode, 
                    5, // Use fewer epochs for comparison
                    &layer_sizes,
                    &activations,
                    learning_rate,
                    loss_function.clone()
                );
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
    
    fn train_single_mode(
        execution_mode: ExecutionMode, 
        epochs: usize, 
        layer_sizes: &[usize],
        activations: &[ActivationType],
        learning_rate: f32,
        loss_function: LossFunction
    ) -> (std::time::Duration, f32) {
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
            
            let mut mlp = MLP::new(
                layer_sizes.to_vec(),
                activations.to_vec(),
                loss_function,
                learning_rate,
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
        training_mode: TrainingMode,
        learning_rate: f32,
        loss_function: LossFunction,
        batch_size: usize,
        layer_sizes: Vec<usize>,
        activations: Vec<ActivationType>,
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
                
                let mut mlp = MLP::new(
                    layer_sizes,
                    activations,
                    loss_function,
                    learning_rate,
                    execution_mode,
                    42
                );
                
                // Train the model
                let train_start = Instant::now();
                
                for epoch in 0..total_epochs {
                    match training_mode {
                        TrainingMode::FullDataset => {
                            mlp.train(&training_images, &training_labels, 1);
                        }
                        TrainingMode::MiniBatch => {
                            // Shuffle and create mini-batches
                            let mut indices: Vec<usize> = (0..training_images.len()).collect();
                            use rand::seq::SliceRandom;
                            let mut rng = rand::thread_rng();
                            indices.shuffle(&mut rng);
                            
                            for chunk in indices.chunks(batch_size) {
                                let batch_images: Vec<Tensor> = chunk.iter()
                                    .map(|&i| training_images[i].clone())
                                    .collect();
                                let batch_labels: Vec<Tensor> = chunk.iter()
                                    .map(|&i| training_labels[i].clone())
                                    .collect();
                                
                                mlp.train(&batch_images, &batch_labels, 1);
                            }
                        }
                        TrainingMode::Stochastic => {
                            // Train on one random example at a time
                            use rand::Rng;
                            let mut rng = rand::thread_rng();
                            for _ in 0..training_images.len() {
                                let idx = rng.gen_range(0..training_images.len());
                                mlp.train(&[training_images[idx].clone()], &[training_labels[idx].clone()], 1);
                            }
                        }
                    }
                    
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
                let final_accuracy = if !testing_images.is_empty() {
                    let mut correct = 0;
                    let mut mlp_guard = mlp_arc.lock().unwrap();
                    if let Some(final_mlp) = mlp_guard.as_mut() {
                        for (test_img, test_label) in testing_images.iter().zip(testing_labels.iter()) {
                            let prediction = final_mlp.forward(test_img);
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
                    }
                    correct as f32 / testing_images.len() as f32 * 100.0
                } else {
                    0.0
                };
                
                sender.send(TrainingProgress {
                    current_epoch: total_epochs,
                    total_epochs,
                    accuracy: final_accuracy,
                    is_complete: true,
                    execution_mode,
                    training_time: Some(duration),
                }).ok();
            }
            Err(e) => {
                sender.send(TrainingProgress {
                    current_epoch: 0,
                    total_epochs,
                    accuracy: 0.0,
                    is_complete: true,
                    execution_mode,
                    training_time: Some(std::time::Duration::from_secs(0)),
                }).ok();
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
        if let Some(mlp) = self.mlp.lock().unwrap().as_mut() {
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
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 1000.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Enhanced MNIST Neural Network Trainer",
        options,
        Box::new(|_cc| Ok(Box::new(MnistApp::default())))
    )
}