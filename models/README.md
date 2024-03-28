# Wine Quality Prediction using Linear Regression

This project aims to predict wine quality based on two features: alcohol content and volatile acidity. It uses a Linear Regression model with feature scaling and polynomial features to achieve more accurate predictions.

# Dataset
The dataset used for this project is the "Red Wine Quality" dataset, which contains various physicochemical properties of red wine and their associated quality ratings.

# Dataset Source: 
Red Wine Quality Dataset

## Requirements

    Python 3.x
    pandas
    numpy
    matplotlib
    scikit-learn

### You can install the required packages using the following command:

pip install pandas numpy matplotlib scikit-learn

# Usage

    - Clone this repository or download the code files.

    - Download the "winequality-red.csv" dataset and place it in the same directory as the code files.

    - Run the "wine_quality_prediction.py" script.

Code Description

    - Import the necessary libraries and load the dataset.

    - Preprocess the data by separating features and the target variable and split the data into training and testing sets.

    - Initialize the Linear Regression model.

    - Perform feature scaling using StandardScaler to standardize the features.

    - Generate polynomial features up to the second degree with interaction terms only using PolynomialFeatures.

    - Combine feature scaling, polynomial feature generation, and linear regression in a pipeline using make_pipeline.

    - Cross-validate the model using the updated feature set to assess its performance.

    - Train the model on the full dataset.

    - Predict wine quality for sample input data (alcohol content and volatile acidity).

# Results

The Linear Regression model's performance is evaluated using R-squared scores from cross-validation. It captures the relationship between alcohol content, volatile acidity, and wine quality.
Conclusion

The Linear Regression model with feature scaling and polynomial features demonstrates a moderate ability to predict wine quality based on alcohol content and volatile acidity. Further exploration of other relevant features and advanced modeling techniques may be required to improve the model's predictive accuracy.



## Author

    Gurpreet Singh 
    
## Acknowledgments

    The dataset used in this project is from the UCI Machine Learning Repository. Special thanks to the data providers.

  
