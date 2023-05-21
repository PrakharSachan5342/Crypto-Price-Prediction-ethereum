# Split the dataset into training and testing sets
#ensure that you replace 'feature1', 'feature2', ... in the code with the actual column names from your dataset that correspond to the features you want to use for prediction

features = ['feature1', 'feature2', ...]  # Replace with the actual column names
target = 'price'  # Replace with the target variable column name

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Make sure the column names you provide in features and target variables match the column names in your dataset.
#
