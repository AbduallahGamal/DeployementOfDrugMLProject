import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
data = pd.read_csv("drug.csv")
print(data.head())

# Converting the non-numeric values into numeric values
data['Sex'] = data['Sex'].map({'M': 1, 'F': 2})
data['BP'] = data['BP'].map({'HIGH': 1, "NORMAL" : 2, "LOW" : 3})
data['Cholesterol'] = data['Cholesterol'].map({'HIGH': 1, "NORMAL" : 2})
data["Drug"] = data["Drug"].map({'DrugA':1, 'DrugB':2, 'DrugC':3, 'DrugX':4,'DrugY':5,
                                 'drugA':1, 'drugB':2, 'drugC':3, 'drugX':4,'drugY':5})

# Splitting the data into training and test datasets
# Select independent and dependent variable
# X data
X = data.drop("Drug", axis=1)
# y data
y = data["Drug"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate the model (Logistic Regression)
regressor = LogisticRegression()

# Fit the model
regressor.fit(X_train, y_train)

# Saving model to disk
# Make pickle file of our model
pickle.dump(regressor, open("model.pkl", "wb"))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[23, 2, 1, 2, 25.234]]))