import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load your own dataset
data = pd.read_csv('models\\winequality-red.csv')

# Apply preprocessing steps
bins = (2, 6.5, 8)
label = ['Bad', 'Good']
data['quality'] = pd.cut(data['quality'], bins=bins, labels=label)

from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data['quality'] = lc.fit_transform(data['quality'])

data.rename(columns={'fixed acidity':'fixed_acidity','volatile acidity':'volatile_acidity',
                     'citric acid':'citric_acid','residual sugar':'residual_sugar',
                     'free sulphur dioxide':'free_sulphur_dioxide','total sulphur dioxide':'total_sulphur_dioxide'
                     }, inplace=True)

# Split the data into features and target variable
X = data.iloc[:, :-1]
y = data['quality']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Model building
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

# Train the model
rf.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)
print(rf_pred[:9])

# Model evaluation
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# Save the model
filename = 'your_model.pickle'
pickle.dump(rf, open(filename, 'wb'))
