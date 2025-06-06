use std::fs::File;
use std::io::{Read, BufReader};
use flate2::read::GzDecoder;


#[derive(Debug)]
pub struct MnistData {
    pub images: Vec<Vec<f64>>,
    pub labels: Vec<u8>,
}

impl MnistData {
    pub fn load_from_files(images_path: &str, labels_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Attempting to load images from: {}", images_path);
        println!("Attempting to load labels from: {}", labels_path);
        
        let images = Self::load_images(images_path)?;
        let labels = Self::load_labels(labels_path)?;
        
        if images.len() != labels.len() {
            return Err(format!("Number of images ({}) and labels ({}) don't match", images.len(), labels.len()).into());
        }
        
        Ok(MnistData { images, labels })
    }
    
    pub fn load_images(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open image file '{}': {}", path, e))?;
        
        let mut reader: Box<dyn Read> = if path.ends_with(".gz") {
            Box::new(GzDecoder::new(file))
        } else {
            Box::new(file)
        };
        
        // Read magic number (should be 2051 for images)
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic number: {}", e))?;
        let magic_num = u32::from_be_bytes(magic);
        if magic_num != 2051 {
            return Err(format!("Invalid magic number for images: {} (expected 2051)", magic_num).into());
        }
        
        // Read number of images
        let mut num_images_bytes = [0u8; 4];
        reader.read_exact(&mut num_images_bytes)
            .map_err(|e| format!("Failed to read number of images: {}", e))?;
        let num_images = u32::from_be_bytes(num_images_bytes) as usize;
        
        // Read image dimensions
        let mut rows_bytes = [0u8; 4];
        let mut cols_bytes = [0u8; 4];
        reader.read_exact(&mut rows_bytes)
            .map_err(|e| format!("Failed to read image rows: {}", e))?;
        reader.read_exact(&mut cols_bytes)
            .map_err(|e| format!("Failed to read image cols: {}", e))?;
        let rows = u32::from_be_bytes(rows_bytes) as usize;
        let cols = u32::from_be_bytes(cols_bytes) as usize;
        
        // println!("Loading {} images of size {}x{}", num_images, rows, cols);
        
        let mut images = Vec::with_capacity(num_images);
        let image_size = rows * cols;
        
        for i in 0..num_images {
            let mut image_data = vec![0u8; image_size];
            reader.read_exact(&mut image_data)
                .map_err(|e| format!("Failed to read image {}: {}", i, e))?;
            
            // Convert to f64 and normalize to [0, 1]
            let image: Vec<f64> = image_data.iter()
                .map(|&pixel| pixel as f64 / 255.0)
                .collect();
            
            images.push(image);
            
            // Progress indicator for every 10000 images
            // if (i + 1) % 10000 == 0 {
            //     println!("Loaded {} images", i + 1);
            // }
        }
        
        Ok(images)
    }
    
    pub fn load_labels(path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open label file '{}': {}", path, e))?;
        
        let mut reader: Box<dyn Read> = if path.ends_with(".gz") {
            Box::new(GzDecoder::new(file))
        } else {
            Box::new(file)
        };
        
        // Read magic number (should be 2049 for labels)
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic number: {}", e))?;
        let magic_num = u32::from_be_bytes(magic);
        if magic_num != 2049 {
            return Err(format!("Invalid magic number for labels: {} (expected 2049)", magic_num).into());
        }
        
        // Read number of labels
        let mut num_labels_bytes = [0u8; 4];
        reader.read_exact(&mut num_labels_bytes)
            .map_err(|e| format!("Failed to read number of labels: {}", e))?;
        let num_labels = u32::from_be_bytes(num_labels_bytes) as usize;
        
        // println!("Loading {} labels", num_labels);
        
        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels)
            .map_err(|e| format!("Failed to read labels: {}", e))?;
        
        Ok(labels)
    }
}