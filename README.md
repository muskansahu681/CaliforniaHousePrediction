I have created the project on "California House Price Prediction"
# California Housing Price Prediction

This project focuses on building a "machine learning model" using Python to estimate real estate values across California. Additionally, a "Power BI dashboard" is designed to visually explore housing data, trends, and patterns.

# Problem Statement

Estimate house prices in California based on variables like:
- Median income
- Housing age
- Population
- Total rooms and bedrooms
- Proximity to the ocean

The model uses "Linear Regression" to predict `MedHouseVal` (median house value), and the results are supported with interactive dashboards in Power BI.

---


---

## ⚙️ Tools and Technologies Used

- **Language:** Python (using Python IDLE)
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- **Visualization:** Power BI
- **Machine Learning Algorithm:** Linear Regression

---

# Implementation Steps

# 1. Import Required Libraries in python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Rest I have loaded the data, performed EDA, Done Feature Selection and Model Building, Done Model Evaluation  in the another filr that I have uploaded as "Muskan Python file.py"

#I have created the dashboard in PowerBI
KPIs: Median House Value, Average Age, Median Income
Visuals: Correlation heatmap, bar charts, filters (like ocean proximity, median income)
Purpose: To visually explore regional housing trends and feature relationships

#I have learned:
How to fetch and work with real-world datasets
Applied Linear Regression for prediction
Performed EDA to gain insights
Built interactive visuals using Power BI
Understood the impact of different features on housing prices








