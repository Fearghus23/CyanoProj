# Cyanobacteria Detection and Counting using EfficientDet

This project uses PyTorch and EfficientDet to detect and count the top 5 most dangerous cyanobacteria from images. It provides a complete pipeline from data preparation, model training, and inference to storing results in a SQL database and displaying them via a web interface.

## Project Structure

- `data/`: Contains training and test images.
- `annotations/`: Stores the annotations file for training.
- `models/`: Stores the trained model weights.
- `database/`: Contains the SQLite database for storing results.
- `templates/`: Contains HTML templates for the Flask app.
- `app.py`: Flask web application.
- `train.py`: Script for training the model.
- `inference.py`: Script for inference and counting.
- `utils.py`: Utility functions for data preparation.
- `dataset.py`: Custom dataset class.
- `model.py`: Model definition.
- `requirements.txt`: Project dependencies.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run `python train.py` to train the model.
4. Run `python inference.py` to perform inference and save counts to the database.
5. Start the Flask application with `python app.py`.
6. Open your browser and go to `http://localhost:5000` to view the results.

## License

This project is licensed under the MIT License.
