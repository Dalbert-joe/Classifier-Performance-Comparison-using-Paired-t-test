# Titanic Survival Prediction using Machine Learning

## Overview

This project focuses on predicting whether a passenger survived the Titanic disaster using multiple machine learning classification algorithms. Various models are trained, evaluated, and compared based on performance metrics and statistical analysis.

---

## Objective

To build and compare different classification models and identify the best-performing model for predicting passenger survival.

---

## Dataset

The dataset used is the Titanic dataset, which contains passenger details such as class, gender, age, fare, and survival status.

### Features

* Survived (Target Variable)
* Pclass (Passenger Class)
* Sex
* Age
* SibSp (Siblings/Spouses aboard)
* Parch (Parents/Children aboard)
* Fare
* Embarked

Irrelevant features like PassengerId, Name, Ticket, and Cabin are removed during preprocessing.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy

---

## Machine Learning Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Machine
* K-Nearest Neighbors
* Naive Bayes (Gaussian)

---

## Workflow

1. Load dataset
2. Data preprocessing
3. Feature encoding
4. Train-test split
5. Model training
6. Model evaluation
7. Cross-validation
8. Graphical analysis
9. Statistical testing (paired t-test)
10. Best model selection
11. Test case prediction

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

---

## Graphs Generated

* Accuracy comparison bar chart
* All metrics comparison graph
* Pairwise accuracy heatmap

---

## Project Structure

```
ML Project
│
├── Classifier.py
├── Titanic-Dataset.csv
├── accuracy.png
├── metrics.png
├── heatmap.png
└── README.md
```

---

## How to Run

1. Install required libraries

```
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

2. Run the Python file

```
python Classifier.py
```

---

## Results

All models are evaluated and compared. The best model is selected based on highest accuracy and tested on sample inputs.

---

## Conclusion

The project demonstrates how machine learning models can effectively predict survival outcomes using real-world data. Proper preprocessing, model comparison, and evaluation techniques play a key role in achieving accurate predictions.

---

## Author

DALBERT JOE J
311124243013

