# Ethan McKerrocher
# 001531184

import pandas as pd
from sklearn import linear_model, metrics
from matplotlib import pyplot

# Initialization of the dataframe.
names = ["fixed-acidity", "volatile-acidity", "citric-acid", "residual-sugar", "chlorides", "free-sulfur-dioxide", "total-sulfur-dioxide", "density", "pH", "sulfates", "alcohol", "quality"]
df = pd.read_csv("data/winequality-white.csv", names=names)

while True:
    # This is the user interface, each option being self-explanatory.
    print("Welcome. Please select an option:")
    print("\n1.) Predict wine quality")
    print("\n2.) View wine data table")
    print("\n3.) View wine data bar graphs")
    print("\n4.) View wine data scatter plots")
    print("\n5.) Quit")
    response = input("> ")
    # Option #1 allows the user to predict the quality of their wine using certain chemical qualities.
    if response == str(1):
        # This initializes and trains the logistic regression model.
        model = linear_model.LogisticRegression(max_iter=10000)
        y = df.values[:, 11]
        x = df.values[:, 0:11]
        model.fit(x, y)

        # y_pred is set to the predictions of every x value in the dataframe in order to be used for measuring the accuracy of the algorithm.
        y_pred = model.predict(x)

        # The algorithm's accuracy is measured and displayed to the user, and then the user inputs their chemical qualities.
        print(f"Please note that this algorithm has an accuracy rate of {metrics.accuracy_score(y, y_pred)}.\nUse it as a supplement for your skills, not a replacement!\n")
        fixed_acidity = input("Please input fixed acidity: ")
        volatile_acidity = input("Please input volatile acidity: ")
        citric_acid = input("Please input citric acid: ")
        residual_sugar = input("Please input residual sugar: ")
        chlorides = input("Please input chlorides: ")
        free_sulfur_dioxide = input("Please input free sulfur dioxide: ")
        total_sulfur_dioxide = input("Please input total sulfur dioxide: ")
        density = input("Please input density: ")
        pH = input("Please input pH: ")
        sulfates = input("Please input sulfates: ")
        alcohol = input("Please input alcohol: ")

        # This prints the predicted quality of the wine based off of the given qualities.
        print(f"Predicted quality: {model.predict([list(map(float, [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulfates, alcohol]))])}")
    # Option #2 allows the user to view the dataset used for the machine learning model as an in-line table.
    elif response == str(2):
        print(df.to_string())
    # Option #3 allows the user to view bar graphs which shows the overall spread of certain chemical qualities of all wines.
    elif response == str(3):
        df.hist()
        pyplot.show()
    # Option #4 shows the user scatter plots of all chemical qualities in relation to each other.
    elif response == str(4):
        pd.plotting.scatter_matrix(df)
        pyplot.show()
    # Option #5 is self-explanatory.
    elif response == str(5):
        exit()
