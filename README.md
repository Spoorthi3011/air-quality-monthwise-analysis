# air-quality-monthwise-analysis
we use advanced data analysis techniques like Random Forest, K-Means, Convolutional Neural Networks (CNN) to monitor monthly air quality Through, you our Access comprehensive." codebases, datasets, and documentation, enabling you to replicate our research and help improve air quality research methodologies.


**Air Quality Analysis with Machine Learning**


**Overview**
This project performs comprehensive air quality analysis using machine learning techniques such as K-Means clustering, Random Forest regression, and Convolutional Neural Networks (CNN). It utilizes the "PRSA_Data_Nongzhanguan" dataset to assess air quality data for various pollutants, including PM2.5, PM10, SO2, NO2, CO, and O3.


**Data Preprocessing**

Data is loaded from a CSV file and selected columns relevant to the analysis.
Missing data is removed.
Data is grouped by month, and an air quality score is calculated for each month based on pollutant concentrations.


**K-Means Clustering**

K-Means clustering is applied to categorize the months into clusters based on their air quality scores.
The month with the best air quality is identified within the cluster with the highest average score.


**Random Forest Regression**

A Random Forest regression model is trained to predict PM2.5 levels based on temperature (TEMP), pressure (PRES), dew point temperature (DEWP), and wind speed (WSPM).
Model performance is evaluated.


**Convolutional Neural Network (CNN)**

A CNN model is built to predict PM2.5 levels.
The model architecture includes convolutional layers, max-pooling, and dense layers.
Model performance is evaluated.


**Results**

The project provides insights into air quality trends and identifies the month with the best air quality.
Random Forest and CNN models are used for air quality prediction.
Model accuracies are reported.
**Usage**
Install the required Python libraries listed in the code.
Download the "PRSA_Data_Nongzhanguan" dataset and specify the file path.
Run the code to perform air quality analysis and prediction.
**Contributors**
Spoorthi

code:
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import calendar

data = pd.read_csv('/content/PRSA_Data_Nongzhanguan_20130301-20170228.csv')
columns = ['month', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
data_selected = data[columns]
data_selected = data_selected.dropna()
grouped_data = data_selected.groupby('month')
pm25_standard = 15.0
pm10_standard = 50.0
so2_standard = 20.0
no2_standard = 40.0
co_standard = 4.0
o3_standard = 60.0
scores = []
for month, month_data in grouped_data:
    avg_pm25 = month_data['PM2.5'].mean()
    avg_pm10 = month_data['PM10'].mean()
    avg_so2 = month_data['SO2'].mean()
    avg_no2 = month_data['NO2'].mean()
    avg_co = month_data['CO'].mean()
    avg_o3 = month_data['O3'].mean()
    score = (avg_pm25 / pm25_standard +
             avg_pm10 / pm10_standard +
             avg_so2 / so2_standard +
             avg_no2 / no2_standard +
             avg_co / co_standard +
             avg_o3 / o3_standard) / 6.0
    scores.append((month, score))
df_scores = pd.DataFrame(scores, columns=['month', 'score'])
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scores[['score']])
cluster_labels = kmeans.labels_
df_scores['cluster'] = cluster_labels
cluster_avg_scores = df_scores.groupby('cluster')['score'].mean()
best_cluster = cluster_avg_scores.idxmax()
best_month_row = df_scores[df_scores['cluster'] == best_cluster].iloc[0]
best_month = best_month_row['month']
best_month_name = calendar.month_name[int(best_month)]

print(df_scores)
print("Month with the Best Air Quality (Cluster", best_cluster, "):", best_month, "-", best_month_name)

best_month_data = data_selected[data_selected['month'] == best_month]
X = best_month_data[['TEMP', 'PRES', 'DEWP', 'WSPM']].values
y = best_month_data['PM2.5'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf = RandomForestRegressor()
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
X_train_cnn = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

cnn = Sequential()
cnn.add(Conv1D(64, 3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
cnn.add(MaxPooling1D(2))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(1))
cnn.compile(optimizer='adam', loss='mse')
cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32)
y_pred_cnn = cnn.predict(X_test_cnn)
rf_accuracy = rf.score(X_test_scaled, y_test)
cnn_accuracy = cnn.evaluate(X_test_cnn, y_test)
print("Random Forest Accuracy: {:.2f}%".format(rf_accuracy * 100))
print("CNN Accuracy: {:.2f}%".format(cnn_accuracy * 100))

