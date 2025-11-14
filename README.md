# Student Performance Factors Prediction

A machine learning project that predicts student exam scores based on various performance factors using four different algorithms.

## Dataset

The dataset (`StudentPerformanceFactors.csv`) contains information about students and their academic performance with the following features:

### Features:
- **Hours_Studied**: Number of hours spent studying
- **Attendance**: Attendance percentage
- **Parental_Involvement**: Level of parental involvement (Low, Medium, High)
- **Access_to_Resources**: Access to educational resources (Low, Medium, High)
- **Extracurricular_Activities**: Participation in extracurricular activities (Yes, No)
- **Sleep_Hours**: Average hours of sleep
- **Previous_Scores**: Previous academic scores
- **Motivation_Level**: Student motivation level (Low, Medium, High)
- **Internet_Access**: Access to internet (Yes, No)
- **Tutoring_Sessions**: Number of tutoring sessions attended
- **Family_Income**: Family income level (Low, Medium, High)
- **Teacher_Quality**: Quality of teaching (Low, Medium, High)
- **School_Type**: Type of school (Public, Private)
- **Peer_Influence**: Influence of peers (Positive, Negative, Neutral)
- **Physical_Activity**: Hours of physical activity per week
- **Learning_Disabilities**: Presence of learning disabilities (Yes, No)
- **Parental_Education_Level**: Education level of parents
- **Distance_from_Home**: Distance from home to school
- **Gender**: Student gender (Male, Female)

### Target Variable:
- **Exam_Score**: The exam score to be predicted

## Machine Learning Models

This project implements four different machine learning algorithms:

1. **Linear Regression**: A simple baseline regression model
2. **Random Forest Regressor**: An ensemble method using multiple decision trees
3. **Support Vector Regression (SVR)**: A non-linear regression model with RBF kernel
4. **Gradient Boosting Regressor**: A sequential ensemble method for regression

## Files

- `StudentPerformanceFactors.csv`: The dataset containing student information and exam scores
- `student_performance_prediction.ipynb`: Jupyter notebook with complete machine learning pipeline
- `README.md`: This file

## Getting Started

1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook student_performance_prediction.ipynb
   ```

3. Run all cells to see the complete analysis and model comparison

## Results

The notebook provides:
- Data exploration and visualization
- Data preprocessing and feature engineering
- Model training and evaluation
- Performance comparison across all four models
- Feature importance analysis
- Predictions vs actual values visualization

## Usage

This project can be used to:
- Understand factors that influence student performance
- Predict exam scores based on student characteristics
- Compare different machine learning approaches for regression tasks
- Analyze feature importance to identify key performance indicators

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter