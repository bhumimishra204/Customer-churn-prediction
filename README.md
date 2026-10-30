# Customer-churn-prediction
An end-to-end Machine Learning project that predicts customer churn using a Gradient Boosting Classifier. The project includes data analysis, model training, evaluation, and an interactive Streamlit web application for generating predictions.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Deployment-Live-success" alt="Deployment">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="Status">
</p>


# Live Demo
**Try the application:** https://customer-churn-prediction-kdsljhrjqglft44gxlulqx.streamlit.app/

# Project Overview
Customer churn occurs when customers stop using a company's products or services. Predicting churn helps businesses identify customers who may leave and take proactive steps to improve customer retention.

This project uses customer data and a trained Machine Learning model to estimate churn risk through an interactive web interface.

# Features 
Churn Prediction: Predict customer churn using a trained Gradient Boosting model.

Interactive Web App: Enter customer details through a Streamlit interface.

Risk Assessment: Display prediction results using the application's configured threshold.

Data Analysis: Explore the dataset and understand patterns related to customer churn.

Model Evaluation: Evaluate the model using classification metrics.

Cloud Deployment: Access the application through a public Streamlit URL.

# Tech Stack
Python:- Core Programming language
Pandas: Data Manipulation and analysis
Numpy:- Numerical computing
Scikit_learn: Model training and evaluation
Streamlit: interactive web application
VS Code: Data exploration and experimentation
Git & GitHub : Version control and project hosting
Streamlit Community Cloud : Application Deployment

# Machine learning Model
Algorithm: Gradient Boosting Classifier

Gradient Boosting combines multiple decision trees in a sequential learning processs. Each new tree attempts to improve the errors made by the previous trees.

## Model Evaluation

The Following results are from the model evaluation previously recorded for this project.

| Metric | Score |
|---|---:|
| Accuracy | 80.62% |
| Precision (Churn class) | 67.72% |
| Recall (Churn class) | 51.60% |
| F1-score (Churn class) | 58.57% |

### Understanding the Results

- **Accuracy:** 80.62% of evaluated predictions were correct.
- **Precision:** 67.72% of customers predicted as likely to churn actually churned.
- **Recall:** The model identified 51.60% of all actual churn customers.
- **F1-score:** 58.57%, balancing precision and recall.
These metrics describe the recorded evaluation and may vary with the dataset split ,model version , and prediction thresold.
## Project Structure

Run the Project Locally
### Prereuisites
Python installed
Git installed
pip package manager

### 1. Clone the repositry
git clone https://github.com/bhumimishra204/Customer-churn-prediction.git
cd Customer-churn-prediction
### 2. Create a virtual environment
### 3. Activate the virtual environment
### 4. Install the required dependencies
### 5. Run the Streamlit application

##  Future Improvements

- Improve churn detection and recall.
- Experiment with Random Forest, XGBoost, and other classification algorithms.
- Perform hyperparameter tuning to improve model performance.
- Add more interactive visualizations.
- Improve the user interface and customer insights.
- Explore Explainable AI (XAI) techniques to understand model predictions.

## Limitations

- Model predictions depend on the quality and characteristics of the training dataset.
- The model may not identify every customer who is likely to churn.
- Predictions should support business decisions rather than be treated as guaranteed outcomes.

##  Author

**Bhumi Mishra**

MCA Student | Machine Learning Enthusiast

- GitHub: [@bhumimishra204](https://github.com/bhumimishra204)

---

## Support

If you find this project useful, consider giving it a ⭐ on GitHub!



