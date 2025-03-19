# Emotion-Based Activity Prediction

## Project Overview

This project studies how people choose activities based on their emotions. We collected data from a survey in the Turkish community, including age (Yaş), gender (Cinsiyet), and job (Meslek) information. The goal is to train a machine learning model that predicts what activity a person might do when they feel happy, sad, or angry.

## File Description

- ActivityRecommendSurvey.csv contains survey result to train model
- learning_file.ipynb is file that i studied to learn how can i observe, train, test model.
- model_builder.ipynb is actual file that i train and test model

## Dataset & Preprocessing

The dataset includes these columns:

- **Age**: A number showing how old the person is.
- **Gender**: A category (Male/Female/Other).
- **Job**: Different job titles (Student, Engineer, Doctor, etc.).
- **Activity When Happy**: What a person likes to do when they are happy.
- **Activity When Sad**: What a person likes to do when they are sad.
- **Activity When Angry**: What a person likes to do when they are angry.
  ![image](https://github.com/user-attachments/assets/1534c743-e2b2-4d9f-8646-2ff941a81eae)

### Data Preprocessing Steps

1. **Cleaning Data**: We removed missing or incorrect data.
2. **Encoding Categorical Data**:
   - Used `OrdinalEncoder` to turn job titles into numbers.
   - Example:
     ```python
     from sklearn.preprocessing import OrdinalEncoder
     ord_enc = OrdinalEncoder()
     df["Job_encoded"] = ord_enc.fit_transform(df[["Job"]])
     df.drop("Job", axis=1, inplace=True)
     ```
3. **Feature Engineering**: Created new columns if needed.

## Model Training

### Approach

- We used a classification model to predict the type of activity a person is likely to do when they experience a particular emotional state.
- Several models were tested, including Decision Trees, Random Forest, and Logistic Regression.

### Training Process

- The dataset was split into training and testing sets using `train_test_split`.
- Model evaluation was done using accuracy, precision, and recall scores.
- We adjusted the model settings (Hyperparameter tuning) to improve model performance.

### Example Code:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
```

![image](https://github.com/user-attachments/assets/6b263023-8665-4f7d-b23e-046d7973ce06)

## Saving the Model

After training the model, we saved it using `joblib.dump()` so we can use it later:

```python
import joblib
joblib.dump(model, 'emotion_activity_model.pkl')
```

This makes it easy to load and use the model without training it again.

## How to Install & Use

### Install Required Libraries

Make sure you have Python and the needed libraries:

```bash
pip install numpy pandas scikit-learn joblib
```

### Run the Model

To use the model:

```python
import joblib
model = joblib.load('emotion_activity_model.pkl')

sample_input = [[21, 0, 2]]  # Example input (age, gender, job)
predicted_activity = model.predict(sample_input)
print(f'Predicted Activity: {predicted_activity[0]}')
```

## Future Improvements

- **More Features**: Add things like stress levels, weather, or social situations.
- **Deep Learning Models**: Experiment with neural networks for better generalization.
- **More Data**: Get data from different people for better predictions.
- **Make It an API**: Turn the model into a web service for real-time predictions.

## Limitations

- The dataset is small, so predictions might not be perfect.
- Not all important factors are included.
- Turning jobs into numbers (Categorical encoding) may not fully show differences between them.

## Conclusion

This project successfully builds a machine learning model to predict people's preferred activities based on their emotional state. While there are areas for improvement, the model demonstrates promising results in understanding human behavior patterns. Future updates can make it even better.
