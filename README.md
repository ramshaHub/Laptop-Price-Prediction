# Laptop Price Prediction

## Overview
This project aims to predict laptop prices based on various features such as brand, RAM, storage, processor type, and customer feedback (ratings and reviews). The model is built using the **RandomForestRegressor** from the `scikit-learn` library and is trained on a dataset of laptops with various specifications.

## Project Structure
- **Data Preprocessing**: Conversion of relevant features to numerical values and feature engineering.
- **Model Training**: A **Random Forest** regression model is trained using a pipeline that preprocesses both numerical and categorical features.
- **Model Evaluation**: After training, the model's performance is evaluated using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the R² score.
- **Visualizations**: The results include visualizations of actual vs predicted prices and feature importance.

## Dependencies
This project requires the following Python libraries:
- `numpy`: For numerical computations
- `pandas`: For data manipulation and analysis
- `scikit-learn`: For machine learning models and preprocessing
- `matplotlib`: For plotting graphs
- `seaborn`: For enhanced visualizations

You can install the required packages using the following command:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Dataset
The dataset used in this project is uploaded from a CSV file, containing various features such as:
- **brand**: The brand of the laptop
- **processor_brand**: The brand of the processor
- **processor_name**: The specific processor model
- **processor_gnrtn**: Processor generation
- **ram_gb**: RAM size in GB
- **ssd**: SSD size in GB
- **hdd**: HDD size in GB
- **graphic_card_gb**: Graphics card memory in GB
- **Number of Ratings**: The total number of ratings the laptop received
- **Number of Reviews**: The total number of reviews the laptop received
- **Price**: The target variable (laptop price)

## Code Breakdown

1. **Loading the Dataset**:  
   The dataset is uploaded using Google Colab's file upload function and is read using `pandas.read_csv()`.

2. **Data Exploration**:  
   Basic exploration of the data is performed using `data.head()` and `data.describe()` to understand the structure and summary statistics of the dataset.

3. **Data Preprocessing**:  
   Several columns such as RAM, SSD, HDD, and Graphics Card memory are converted from strings (e.g., '16 GB') to numerical values. The total storage is calculated by adding SSD and HDD storage.

4. **Feature Engineering**:  
   Additional feature `total_storage` is created by summing the SSD and HDD capacities.

5. **Preprocessing for Numerical and Categorical Features**:  
   - Numerical features are scaled using `StandardScaler`.
   - Categorical features are one-hot encoded using `OneHotEncoder`.

6. **Model Training (Random Forest Regressor)**:  
   A **Random Forest Regressor** model is built to predict laptop prices. The model is included in a pipeline that handles both preprocessing and training.

7. **Model Evaluation**:  
   The model is evaluated on the test set using the following metrics:
   - **Mean Squared Error (MSE)**
   - **Root Mean Squared Error (RMSE)**
   - **Mean Absolute Error (MAE)**
   - **R² Score**

8. **Visualizations**:
   - A scatter plot is created to compare the actual vs predicted prices.
   - A bar chart is generated to show the importance of features in predicting the laptop prices.

## Model Evaluation

The following metrics are used to evaluate the model:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Square root of MSE, providing an interpretable metric in the same unit as the target variable.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
- **R² Score**: Indicates how well the model fits the data, with a value of 1 representing a perfect fit.

## Example Output

```plaintext
Model Evaluation:
Mean Squared Error: 4500000.00
Root Mean Squared Error: 2121.32
Mean Absolute Error: 1600.25
R² Score: 0.85
```

### Visualizations

1. **Actual vs Predicted Prices**:
   A scatter plot compares the actual laptop prices with the predicted ones. Ideally, the points should align along a diagonal line.

2. **Feature Importances**:
   A bar plot shows the relative importance of features in predicting laptop prices, based on the Random Forest model.

## Usage

1. **Upload your dataset**: 
   Ensure the dataset has columns matching the expected features.
2. **Run the code**: 
   The model will preprocess the data, train a Random Forest regressor, and display evaluation metrics along with visualizations.
3. **Interpret the results**: 
   Use the model evaluation metrics and feature importances to understand the performance and insights from the model.

## Conclusion

This project demonstrates how to build and evaluate a machine learning model for predicting laptop prices based on various hardware specifications and customer feedback. The pipeline approach ensures that preprocessing and model training steps are streamlined for better model performance and evaluation.
