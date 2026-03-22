# Petroleum Samples Classification using Deep Learning

## Project Overview
This project focuses on the automated classification of petroleum samples based on their chemical and physical properties. Using a Deep Learning approach, the model identifies specific categories of petroleum, which has significant applications in industrial automation and quality control within the energy sector.

## Documentation
A detailed technical report (in Polish) covering the research methodology, data stability analysis, and performance comparison of various classifiers is available here: [Project Report](./Petroleum_Classification_Project_Report.pdf)

## Key Features
* **Deep Learning Architecture:** Implemented a Multi-Layer Perceptron (MLP) using **TensorFlow/Keras**.
* **Data Preprocessing:** Performed feature scaling (StandardScaler) and handled complex scientific datasets to ensure high model stability.
* **Performance Monitoring:** Visualized training history (Accuracy/Loss) to optimize hyperparameters and prevent overfitting.
* **Scalability:** The solution is designed to handle multi-class classification tasks with high-dimensional input data.

## Tech Stack
* **Language:** Python
* **IDE:** PyCharm (Professional Project Structure)
* **Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib.
* **Environment:** Local Python Virtual Environment (venv).

## How to Run
1. **Clone the repository:** `git clone https://github.com/Hyp3reeek/PetroleumSamplesClasification.git`
2. **Open in PyCharm:** Open the project folder in PyCharm.
3. **Setup Virtual Environment:** PyCharm will usually prompt you to create a virtual environment (venv) automatically.
4. **Install Dependencies:** Run `pip install tensorflow pandas scikit-learn matplotlib` in the PyCharm terminal.
5. **Run the application:** Execute the main script (e.g., `python main.py`) to start the data processing and model training.

---
*This project was developed as part of my academic work at Gdańsk University of Technology, focusing on practical applications of Machine Learning in data-heavy industries.*
