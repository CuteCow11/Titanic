This project builds a machine learning model to predict whether a passenger survived the Titanic disaster, based on features like age, sex, ticket class, and fare. We use logistic regression for classification and visualize the data to understand key patterns.

What the Code Does?
1.Loads the Titanic dataset (titanic_train.csv)
2.Explores and visualizes missing data and relationships between features and survival
3.Cleans the data by handling missing values and dropping irrelevant columns
4.Converts categorical variables (like gender and embarkation port) into numeric format for modeling
5.Splits the data into training and testing sets
6.Trains a logistic regression model to predict survival
7.Evaluates the model's performance using a confusion matrix and accuracy score

Key Libraries Used
1.pandas and numpy: Data manipulation and numerical operations
2.matplotlib and seaborn: Data visualization
3.scikit-learn: Machine learning model, data splitting, and evaluation metrics
