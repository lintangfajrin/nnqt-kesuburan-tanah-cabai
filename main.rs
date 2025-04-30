mod data;
mod model;
mod utils;

use std::error::Error;
use data::{load_csv, normalize_features, split_data};
use model::NeuralNetwork;
use utils::plot_accuracy;

fn main() -> Result<(), Box<dyn Error>> {
    // Path to data files with correct folder structure
    let csv_path = "data/chili_soil_data.csv";

    // Try to load data from existing CSV
    let data = match load_csv(csv_path) {
        Ok(loaded_data) => {
            println!("Successfully loaded data from {}", csv_path);
            loaded_data
        }
        Err(e) => {
            println!("Error loading CSV file: {}", e);
            println!("Please ensure the file 'data/chili_soil_data.csv' exists and is accessible.");
            return Err(Box::new(e));
        }
    };

    println!("Loaded {} data points", data.len());

    // Normalize the data
    let mut data_normalized = data.clone();
    normalize_features(&mut data_normalized);

    // Split into training and testing sets (80% training, 20% testing)
    let (train_data, test_data) = split_data(&data_normalized, 0.8);

    println!("Training data size: {}", train_data.len());
    println!("Testing data size: {}", test_data.len());

    // Create neural network
    let input_size = 4; // Temperature, Humidity, Soil_pH, Soil_Moisture
    let hidden_size = 8; // Number of neurons in hidden layer
    let output_size = 1; // Fertile (0 or 1)
    let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size);

    // Train the neural network
    let epochs = 1000;
    let learning_rate = 0.01;
    println!("Starting training for {} epochs...", epochs);
    let accuracy_history = nn.train(&train_data, epochs, learning_rate);

    // Evaluate on test data
    let test_accuracy = nn.evaluate(&test_data);
    println!("Final test accuracy: {:.2}%", test_accuracy * 100.0);

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    // Plot accuracy vs epochs
    plot_accuracy(&accuracy_history, "output/accuracy_vs_epochs.png")?;

    // Make some predictions
    println!("\nSample predictions:");
    for i in 0..5 {
        if i < test_data.len() {
            let features = &test_data[i].features;
            let actual = test_data[i].label;
            let prediction = nn.predict(features);

            println!(
                "Features: Temperature={:.1}, Humidity={:.1}, pH={:.2}, Moisture={:.2} | Actual: {}, Predicted: {}",
                features[0], features[1], features[2], features[3],
                actual, prediction
            );
        }
    }

    Ok(())
}