import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

# Get the California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Create a DataFrame
df = housing.frame

# Look at the data
print(df.head())

# Number of rows and columns
print(df.shape)

# General information about the DataFrame
print(df.info())

# Statistical summary
print(df.describe())


# Create a scatter plot
plt.scatter(df["Longitude"], df["Latitude"])

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing")

plt.show()
