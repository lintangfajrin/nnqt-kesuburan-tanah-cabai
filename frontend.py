import sys
import os
import csv
import random
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# DataPoint class
class DataPoint:
    def __init__(self, features, label):
        self.features = features
        self.label = label

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size=4, hidden_size=8, output_size=1):
        self.weights1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.weights2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.bias1 = np.zeros((hidden_size, 1))
        self.bias2 = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1.0 - s)

    def forward(self, inputs):
        inputs = np.array(inputs).reshape(-1, 1)
        hidden = self.sigmoid(self.weights1 @ inputs + self.bias1)
        output = self.sigmoid(self.weights2 @ hidden + self.bias2)
        return hidden, output

    def train(self, data, epochs, learning_rate, callback=None):
        accuracy_history = []
        loss_history = []
        for epoch in range(epochs):
            correct = 0
            total_loss = 0.0
            for point in data:
                hidden, output = self.forward(point.features)
                error = output - point.label
                # Calculate MSE loss
                loss = float(np.mean(error ** 2))
                total_loss += loss
                d_output = error * self.sigmoid_derivative(output)
                self.weights2 -= learning_rate * (d_output @ hidden.T)
                self.bias2 -= learning_rate * d_output
                d_hidden = (self.weights2.T @ d_output) * self.sigmoid_derivative(hidden)
                self.weights1 -= learning_rate * (d_hidden @ np.array(point.features).reshape(1, -1))
                self.bias1 -= learning_rate * d_hidden
                if (output[0, 0] > 0.5) == point.label:
                    correct += 1
            accuracy = correct / len(data)
            avg_loss = total_loss / len(data)
            accuracy_history.append(accuracy)
            loss_history.append(avg_loss)
            if callback:
                callback(epoch + 1, accuracy, avg_loss)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}: Accuracy = {accuracy * 100:.2f}%, Loss = {avg_loss:.4f}")
        return accuracy_history, loss_history

    def predict(self, features):
        _, output = self.forward(features)
        return 1 if output[0, 0] > 0.5 else 0

    def evaluate(self, data):
        correct = sum(1 for point in data if self.predict(point.features) == point.label)
        return correct / len(data)

# Data handling functions
def load_csv(path):
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 5:
                features = [float(x) for x in row[:4]]
                label = int(row[4])
                data.append(DataPoint(features, label))
    return data

def normalize_features(data):
    features = np.array([point.features for point in data]).T
    min_vals, max_vals = features.min(axis=1), features.max(axis=1)
    for point in data:
        for i in range(4):
            range_val = max_vals[i] - min_vals[i]
            point.features[i] = (point.features[i] - min_vals[i]) / range_val if range_val > 0 else 0.5
    return min_vals, max_vals

def split_data(data, train_ratio):
    train_data, test_data = [], []
    for point in data:
        if random.random() < train_ratio:
            train_data.append(DataPoint(point.features.copy(), point.label))
        else:
            test_data.append(DataPoint(point.features.copy(), point.label))
    return train_data, test_data

# Plotting class
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(111)
        self.axes2 = self.axes1.twinx()  # Secondary y-axis for loss
        super().__init__(fig)
        self.setParent(parent)
        self.accuracy_data = []
        self.loss_data = []
        self.epochs = []

    def update_plot(self, epoch, accuracy, loss):
        self.epochs.append(epoch)
        self.accuracy_data.append(accuracy)
        self.loss_data.append(loss)
        self.axes1.clear()
        self.axes2.clear()
        self.axes1.plot(self.epochs, self.accuracy_data, 'r-', label='Accuracy')
        self.axes2.plot(self.epochs, self.loss_data, 'b-', label='Loss')
        self.axes1.set_xlabel('Epochs')
        self.axes1.set_ylabel('Accuracy', color='r')
        self.axes2.set_ylabel('Loss', color='b')
        self.axes1.set_title('Training Progress')
        self.axes1.tick_params(axis='y', colors='r')
        self.axes2.tick_params(axis='y', colors='b')
        self.axes1.grid(True)
        lines1, labels1 = self.axes1.get_legend_handles_labels()
        lines2, labels2 = self.axes2.get_legend_handles_labels()
        self.axes1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        self.draw()

    def reset(self):
        self.epochs = []
        self.accuracy_data = []
        self.loss_data = []
        self.axes1.clear()
        self.axes2.clear()
        self.draw()

# Main GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soil Fertility Predictor")
        self.setGeometry(100, 100, 700, 600)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)

        # Apply stylesheet for professional look
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #f0f4f8, stop:1 #d9e2ec);
            }
            QLabel {
                font: 14px 'Segoe UI';
                color: #333;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 5px;
                font: 14px 'Segoe UI';
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
                font: 14px 'Segoe UI';
            }
            QPushButton:hover {
                background-color: #005ba1;
            }
            QPushButton:pressed {
                background-color: #004080;
            }
        """)

        # CSV loading
        self.csv_layout = QHBoxLayout()
        self.csv_layout.setSpacing(10)
        self.csv_input = QLineEdit(self)
        self.csv_input.setPlaceholderText("Select or enter CSV file path")
        self.browse_btn = QPushButton("Browse", self)
        self.browse_btn.clicked.connect(self.browse_csv)
        self.load_btn = QPushButton("Load Data", self)
        self.load_btn.clicked.connect(self.load_data)
        self.csv_layout.addWidget(QLabel("CSV File:"))
        self.csv_layout.addWidget(self.csv_input)
        self.csv_layout.addWidget(self.browse_btn)
        self.csv_layout.addWidget(self.load_btn)
        self.layout.addLayout(self.csv_layout)

        # Training parameters
        self.train_layout = QHBoxLayout()
        self.train_layout.setSpacing(10)
        self.epochs_input = QLineEdit("1000", self)
        self.epochs_input.setMaximumWidth(100)
        self.lr_input = QLineEdit("0.001", self)
        self.lr_input.setMaximumWidth(100)
        self.train_btn = QPushButton("Train Model", self)
        self.train_btn.clicked.connect(self.train_model)
        self.train_layout.addWidget(QLabel("Epochs:"))
        self.train_layout.addWidget(self.epochs_input)
        self.train_layout.addWidget(QLabel("Learning Rate:"))
        self.train_layout.addWidget(self.lr_input)
        self.train_layout.addWidget(self.train_btn)
        self.train_layout.addStretch()
        self.layout.addLayout(self.train_layout)

        # Training status
        self.status_layout = QHBoxLayout()
        self.status_layout.setSpacing(10)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font: bold 14px 'Segoe UI';")
        self.epoch_label = QLabel("Epoch: 0")
        self.loss_label = QLabel("Loss: 0.0000")
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.epoch_label)
        self.status_layout.addWidget(self.loss_label)
        self.status_layout.addStretch()
        self.layout.addLayout(self.status_layout)

        # Plot
        self.canvas = PlotCanvas(self, width=6, height=4, dpi=100)
        self.layout.addWidget(self.canvas)

        # Sample predictions
        self.pred_label = QLabel("Sample Predictions: None")
        self.pred_label.setStyleSheet("font: 12px 'Consolas'; color: #333; background: #fff; padding: 10px; border: 1px solid #ccc; border-radius: 5px;")
        self.pred_label.setWordWrap(True)
        self.layout.addWidget(self.pred_label)

        self.layout.addStretch()

        self.data = None
        self.nn = None
        self.test_data = None
        self.min_vals = None
        self.max_vals = None

    def browse_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.csv_input.setText(file_path)

    def load_data(self):
        path = self.csv_input.text()
        if not os.path.exists(path):
            QMessageBox.critical(self, "Error", "CSV file not found!")
            return
        try:
            self.data = load_csv(path)
            self.status_label.setText(f"Status: Loaded {len(self.data)} points")
            self.epoch_label.setText("Epoch: 0")
            self.loss_label.setText("Loss: 0.0000")
            self.pred_label.setText("Sample Predictions: None")
            self.canvas.reset()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")
            self.status_label.setText("Status: Error")

    def train_model(self):
        if not self.data:
            QMessageBox.critical(self, "Error", "Load data first!")
            return
        try:
            epochs = int(self.epochs_input.text())
            lr = float(self.lr_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid epochs or learning rate!")
            return

        self.status_label.setText("Status: Training...")
        self.canvas.reset()
        QApplication.processEvents()

        def update_ui(epoch, accuracy, loss):
            self.epoch_label.setText(f"Epoch: {epoch}")
            self.loss_label.setText(f"Loss: {loss:.4f}")
            self.canvas.update_plot(epoch, accuracy, loss)
            QApplication.processEvents()

        data_copy = [DataPoint(point.features.copy(), point.label) for point in self.data]
        self.min_vals, self.max_vals = normalize_features(data_copy)
        train_data, self.test_data = split_data(data_copy, 0.8)
        self.nn = NeuralNetwork()
        accuracy_history, loss_history = self.nn.train(train_data, epochs, lr, update_ui)
        test_accuracy = self.nn.evaluate(self.test_data)
        self.status_label.setText(f"Status: Done. Test accuracy: {test_accuracy*100:.2f}%")

        # Display sample predictions
        pred_text = "Sample Predictions:\n"
        pred_text += "Temp  Hum  pH  Moist  Actual  Predicted\n"
        pred_text += "-" * 40 + "\n"
        for i in range(min(5, len(self.test_data))):
            point = self.test_data[i]
            # Denormalize features for display
            features = [
                point.features[j] * (self.max_vals[j] - self.min_vals[j]) + self.min_vals[j]
                for j in range(4)
            ]
            pred = self.nn.predict(point.features)
            pred_text += (f"{features[0]:<5.1f} {features[1]:<4.1f} {features[2]:<3.1f} "
                         f"{features[3]:<6.1f} {'Fertile' if point.label else 'Not Fertile':<7} "
                         f"{'Fertile' if pred else 'Not Fertile'}\n")
        self.pred_label.setText(pred_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())