# AI Powered Edge-Based Gunshot Detection for Protecting Wildlife

This repository contains the complete implementation of an AI-powered edge-based gunshot detection system designed for wildlife protection. The project combines machine learning, hardware implementation, and real-time audio processing to detect gunshots in wildlife environments.

## Repository Structure

### `hardware/`
Contains hardware-related files, logs, and performance metrics for the edge device implementation.

#### `hardware/LOG/`
- `Receiver_node_log.txt` - Log file for the receiver node operations and data transmission
- `Sensor_node_log.txt` - Log file for the sensor node audio capture and preprocessing

#### `hardware/Enclosure`
Directory containing test results for the physical enclosure.

- `Axial force displace.xlsx` - Excel file documenting axial force displacement test results.
- `S-N test.xlsx` - Excel file documenting S-N (Stress-Number) test results.
- `tension test.xlsx` - Excel file documenting tension test results.

### `ML/`
Contains all machine learning related files including datasets, training scripts, models, and performance analysis.

#### `ML/dataset/`
Audio dataset used for training and testing the gunshot detection model.

- `environment_audio/` - Directory containing environmental/background audio samples (non-gunshot sounds)
- `gunshots_audio/` - Directory containing gunshot audio samples for training

#### `ML/figures_data/`
CSV files containing data used to generate performance plots and visualizations.

**Accuracy Plot Data:**
- `accuracy_plot_data__400.csv` - Training/validation accuracy data for 400-sample dataset
- `accuracy_plot_data__1000.csv` - Training/validation accuracy data for 1000-sample dataset
- `accuracy_plot_data__1700.csv` - Training/validation accuracy data for 1700-sample dataset
- `accuracy_plot_data__2000.csv` - Training/validation accuracy data for 2000-sample dataset

**Confusion Matrix Data:**
- `confusion_matrix_data__400.csv` - Confusion matrix results for 400-sample dataset
- `confusion_matrix_data__1000.csv` - Confusion matrix results for 1000-sample dataset
- `confusion_matrix_data__1700.csv` - Confusion matrix results for 1700-sample dataset
- `confusion_matrix_data__2000.csv` - Confusion matrix results for 2000-sample dataset

**Loss Plot Data:**
- `loss_plot_data__400.csv` - Training/validation loss data for 400-sample dataset
- `loss_plot_data__1000.csv` - Training/validation loss data for 1000-sample dataset
- `loss_plot_data__1700.csv` - Training/validation loss data for 1700-sample dataset
- `loss_plot_data__2000.csv` - Training/validation loss data for 2000-sample dataset

#### `ML/metrics/`
Comprehensive performance metrics and statistical analysis results.

**Complete Study Metrics:**
- `complete_study_metrics__400.json` - Complete evaluation metrics for 400-sample study
- `complete_study_metrics__1000.json` - Complete evaluation metrics for 1000-sample study
- `complete_study_metrics__1700.json` - Complete evaluation metrics for 1700-sample study
- `complete_study_metrics__2000.json` - Complete evaluation metrics for 2000-sample study

**Statistical Summaries:**
- `statistical_summary__400.csv` - Statistical summary for 400-sample dataset
- `statistical_summary__1000.csv` - Statistical summary for 1000-sample dataset
- `statistical_summary__1700.csv` - Statistical summary for 1700-sample dataset
- `statistical_summary__2000.csv` - Statistical summary for 2000-sample dataset

**Training History:**
- `training_history_per_epoch__400.csv` - Epoch-wise training history for 400-sample dataset
- `training_history_per_epoch__1000.csv` - Epoch-wise training history for 1000-sample dataset
- `training_history_per_epoch__1700.csv` - Epoch-wise training history for 1700-sample dataset
- `training_history_per_epoch__2000.csv` - Epoch-wise training history for 2000-sample dataset

#### `ML/training/`
Model training scripts and trained model files.

- `model_training_quantization_jupyter.ipynb` - Jupyter notebook for model training and quantization
- `gunshot_model_quant.tflite` - Quantized TensorFlow Lite model for edge deployment

#### `ML/training/exported_models/`
Directory containing exported model files in different formats.

##### `ML/training/exported_models/keras/`
- `gunshot_detection.keras` - Full Keras model file with complete architecture and weights

##### `ML/training/exported_models/quantized/`
- `gunshot_model_quant.tflite` - Quantized TensorFlow Lite model optimized for edge devices

