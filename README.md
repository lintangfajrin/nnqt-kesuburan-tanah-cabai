Chili Soil Fertility Prediction using Neural Network in Rust 🌶️

This project is a Rust-based machine learning application that predicts the fertility of soil for chili cultivation using neural network. The model is trained using environmental data such as:
- Temperature
- Humidity
- Soil pH
- Soil Moisture

It classifies the soil as **fertile (1)** or **not fertile (0)**.

## Features

- Feedforward Neural Network built from scratch in Rust
- Custom CSV data loader
- Feature normalization
- 80/20 train-test split
- Accuracy tracking over epochs
- Visualization of accuracy using line plot
- CLI output of sample predictions

## Project Structure
. ├── data/ │ └── chili_soil_data.csv # Your input dataset ├── output/ │ └── accuracy_vs_epochs.png # Accuracy plot after training ├── src/ │ ├── data.rs # CSV reading, normalization, train-test split │ ├── model.rs # Neural network logic │ ├── utils.rs # Utility functions (e.g. plot) │ └── main.rs # Main program entry point ├── Cargo.toml └── README.md
