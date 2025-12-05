# Student Performance Factors Prediction

A comprehensive comparison of **Traditional Machine Learning vs Deep Learning** algorithms for predicting student exam scores based on various performance factors.

This project implements **6 different algorithms** (3 Traditional ML + 3 Deep Learning) to demonstrate the strengths and capabilities of both approaches in educational data analysis.

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
- **Exam_Score**: The exam score to be predicted (continuous value)

## Algorithms Implemented

### Traditional Machine Learning (3 Models)
1. **Linear Regression**: Simple baseline model establishing linear relationships between features and exam scores
2. **Support Vector Regression (SVR)**: Non-linear regression using RBF kernel to capture complex patterns
3. **Gradient Boosting Regressor**: Sequential ensemble method that builds trees iteratively to minimize prediction errors

### Deep Learning (3 Models)
4. **Wide & Deep Neural Network**: Hybrid architecture combining:
   - **Wide path**: Linear layer for memorization of feature interactions
   - **Deep path**: Multi-layer network (64→32 neurons) for generalization
   - Designed to balance learning specific patterns and general representations

5. **ResNet (Residual Network)**: Deep neural network with skip connections:
   - Enables training of deeper networks by addressing vanishing gradient problem
   - Residual connections allow gradient flow through multiple layers
   - Better feature extraction through depth

6. **Bayesian Neural Network**: Probabilistic deep learning approach:
   - Provides uncertainty quantification in predictions
   - Multiple forward passes to estimate prediction confidence
   - Useful for identifying when the model is uncertain about predictions

## Project Structure

```
StudentPerformanceFactors/
│
├── StudentPerformanceFactors.csv          # Dataset (6,609 student records)
├── student.ipynb                          # Main Jupyter notebook with all implementations
└── README.md                              # Project documentation
```

## Key Features

✨ **Comprehensive Model Comparison**: Side-by-side evaluation of Traditional ML vs Deep Learning approaches  
📊 **Extensive Data Analysis**: Exploratory data analysis with correlation analysis (numerical and categorical variables)  
🎯 **Multiple Evaluation Metrics**: RMSE, MAE, R² Score, and custom Accuracy (±2 points tolerance)  
📈 **Visualization Suite**: 14+ professional plots including feature importance, learning curves, and model comparisons  
🔬 **Advanced Techniques**: Includes Cramér's V for categorical associations and correlation ratio (η) analysis

## Getting Started

### Prerequisites

Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter scipy
```

**Required packages:**
- `pandas`, `numpy`: Data manipulation and numerical operations
- `matplotlib`, `seaborn`: Data visualization
- `scikit-learn`: Traditional ML algorithms and preprocessing
- `tensorflow`: Deep learning framework (for Wide & Deep, ResNet, Bayesian NN)
- `scipy`: Statistical functions (for correlation analysis)

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StudentPerformanceFactors.git
   cd StudentPerformanceFactors
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook student.ipynb
   ```

3. Execute all cells sequentially to:
   - Load and explore the dataset
   - Perform comprehensive EDA with statistical analysis
   - Train all 6 models (3 ML + 3 DL)
   - Compare performance metrics
   - Generate visualizations and insights

## Results & Analysis

### What the Notebook Provides:

#### 1. **Exploratory Data Analysis**
- Distribution analysis of 19 features and target variable
- Correlation heatmaps (numerical variables)
- Mixed correlation matrix using Cramér's V and correlation ratio (η)
- Categorical variable relationships with exam scores

#### 2. **Model Training & Evaluation**
- **Traditional ML Models**: Linear Regression, SVR, Gradient Boosting
- **Deep Learning Models**: Wide & Deep, ResNet, Bayesian Neural Network
- Training with optimized hyperparameters
- Validation and testing on separate datasets

#### 3. **Performance Metrics**
Each model evaluated using:
- **RMSE** (Root Mean Square Error): Overall prediction error
- **MAE** (Mean Absolute Error): Average prediction deviation
- **R² Score**: Proportion of variance explained
- **Accuracy (±2 points)**: Percentage of predictions within 2 points of actual score

#### 4. **Visualizations Generated** (14+ plots)
- Exam score distribution
- Numerical variables distributions
- Categorical-exam score correlations
- Correlation heatmaps (numerical and mixed)
- Prediction vs Actual scatter plots (all models)
- Model comparison charts (RMSE, MAE, R², Accuracy)
- Feature importance analysis
- Learning curves (Deep Learning models)
- Ultimate comparison: Traditional ML vs Deep Learning

#### 5. **Feature Importance**
Identification of top predictive features from Gradient Boosting model to understand key factors influencing student performance.

## Use Cases

This project demonstrates:
- 🎓 **Educational Analytics**: Identify factors influencing student success
- 🤖 **ML vs DL Comparison**: Understand when to use traditional ML vs deep learning
- 📊 **Regression Modeling**: Complete pipeline from EDA to model deployment
- 🔍 **Feature Engineering**: Handling mixed data types (categorical + numerical)
- 📈 **Model Selection**: Evidence-based approach to choosing the right algorithm

## Performance Insights

> **Note**: Based on the project implementation, you'll find detailed comparisons showing:
> - Which approach (ML vs DL) performs better for this dataset size and complexity
> - Trade-offs between model complexity and performance
> - Uncertainty quantification from Bayesian Neural Network
> - The impact of different architectures on prediction accuracy

## Technical Requirements

- **Python**: 3.7 or higher
- **Core Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Deep Learning**: tensorflow (Keras API)
- **Statistics**: scipy
- **Environment**: Jupyter Notebook or JupyterLab

## Dataset Statistics

- **Total Records**: 6,609 students
- **Features**: 19 (13 categorical + 6 numerical)
- **Target**: Exam_Score (continuous variable)
- **No missing values** in target variable (cleaned dataset)