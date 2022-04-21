import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import seaborn as sns
sns.set()

unprocessed_data = pd.read_excel('kenya-accidents-database.xlsx')
# print(unprocessed_data)

# Cleaning the data
unprocessed_data1 = unprocessed_data.drop(['Unnamed: 12', 'PLACE', 'BASE/SUB BASE', 'BRIEF ACCIDENT DETAILS', 'Date DD/MM/YYYY', ], axis=1)
unprocessed_data2 = unprocessed_data1.sort_values('TIME 24 HOURS')
unprocessed_dataframe = pd.DataFrame(unprocessed_data2)
processed_dataframe = unprocessed_dataframe.dropna()

# print(unprocessed_dataframe.shape)
# print(unprocessed_dataframe.shape)
# print(processed_dataframe.to_string())

# print(processed_dataframe.columns.values)

input_columns = ['TIME 24 HOURS', 'AGE', 'CAUSE CODE', 'NO.']
# print(input_columns)

X_input = processed_dataframe[input_columns].to_numpy()
# print(X_input)

scaler = preprocessing.StandardScaler().fit(X_input)
X_scaled = scaler.transform(X_input)
# print(X_scaled)

# checking for elboes method for the best value for 'n', for clustering
wcss = []
n = range(1, 10)
for i in n:
    kmeans = KMeans(i)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(n, wcss)
plt.show()

# from the plot, n = 5 works okay, fit the model with 5 clusters

kmeans = KMeans(5)
kmeans.fit(X_scaled)

predict = kmeans.predict(X_scaled)

# check if the output number is the same as input
print(processed_dataframe.shape)
print(len(predict))
print(len(X_input))

processed_dataframe['cluster'] = predict
# print(processed_dataframe)

# understanding the clusters
# print("Victims in Cluster 0: ", processed_dataframe[processed_dataframe['cluster'] == 0]['VICTIM'].unique())
# print("Victims in Cluster 1: ", processed_dataframe[processed_dataframe['cluster'] == 1]['VICTIM'].unique())
# print("Victims in Cluster 2: ", processed_dataframe[processed_dataframe['cluster'] == 2]['VICTIM'].unique())
# print("Victims in Cluster 3: ", processed_dataframe[processed_dataframe['cluster'] == 3]['VICTIM'].unique())
# print("Victims in Cluster 4: ", processed_dataframe[processed_dataframe['cluster'] == 4]['VICTIM'].unique())

motorcyclist = 0
driver = 1
pedestrian = 2
passenger = 3
p_passenger = 4

labels = ['Motorcycle', 'Driver', 'Pedestrian', 'Passenger', 'P_passenger']

mats = input_columns

cnt = 0
plt.figure(figsize=(100, 100))
for i in range (len(mats)):
    for j in range(len(mats)):
        if i < j:
            cnt = cnt + 1
            plt.subplot(2, 3, cnt)
            x = processed_dataframe[mats[j]].to_numpy().astype(float)
            y = processed_dataframe[mats[i]].to_numpy().astype(float)
            scatter = plt.scatter(x, y, c=predict, cmap='rainbow')
            plt.xlabel(mats[j])
            plt.ylabel(mats[i])
            plt.legend(handles=scatter.legend_elements()[0], labels=labels, title="classes")

plt.show()





