import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

dataset = pd.read_csv("IMDB_movie_reviews_details.csv")
#print(dataset) #print csv values
print(dataset.info())
df = dataset.drop(['id'], axis = 1)

#print(df)
df['year'] = df['year'].str.replace('I', '')
df['year'] = df['year'].str.replace(' ', '')
df['year'] = pd.to_numeric(df['year'])  #fixing the data according to our need
#print(df.info())
df['votes'] = df['votes'].str.replace(',', '')
df['votes'] = pd.to_numeric(df["votes"])
df['gross'] = df['gross'].str.replace('$', '')
df['gross'] = df['gross'].str.replace('M', '')
df['gross'] = pd.to_numeric(df["gross"])
#print(df)
#now we are going to engineer the features to make them viable for the model
df[["genre_1","genre_2","genre_3"]] = df['genre'].str.split(',', n = 3, expand=True)
df = df.drop(['genre'], axis=1)
df['genre_1'] = df['genre_1'].str.replace(' ', '')
df['genre_2'] = df['genre_2'].str.replace(' ', '')
df['genre_3'] = df['genre_3'].str.replace(' ', '')
listofzeros = [0] * 1000
#df.head() #initialize the values with zeroes first
l1 = df.genre_1.unique()
l2 = df.genre_2.unique()
l3 = df.genre_3.unique()
l = list(l1) + list(l2) + list(l3)
l = [i for i in l if i]
l = list(set(l))
for genre in l:
    df[genre] = listofzeros
for genre in l:
    for x in range(1000):
        if df.at[x, 'genre_1'] == genre or df.at[x, 'genre_2'] == genre or df.at[x, 'genre_3'] == genre:
            df.at[x, genre] = 1
#plt.plot(df['rating'], df['gross'], marker = 'o')
#plt.show()
df_model = df[['year', 'runtime', 'votes', 'metascore', 'gross', 'Mystery', 'Drama', 'Musical', 'Fantasy', 'Adventure', 'Western', 'Thriller', 'War', 'Biography', 'Family', 'Sport', 'Film-Noir', 'Music', 'Sci-Fi', 'Animation', 'Romance', 'Crime', 'Action', 'Comedy', 'History', 'rating']]
corral = sns.scatterplot(x = 'gross', y = 'rating', data = df_model)
corral.set_title("Gross Revenue vs Ratings")
plt.show()
X = df_model.iloc[:,:-1].values
Y = df_model.iloc[:,25].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,3:5])
X[:,3:5] = imputer.transform(X[:,3:5]) # imputes the missing values in X with mean data(default)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
#Training and testing data has been made!
tldr = sns.heatmap(df_model.corr(), annot = True)
plt.show()
#After the data has been refined for machine learning usage, we will use 2 different
#algorithms to predict imdb scores of various movies as well as give accuracy
#of each algorithm

#1. Linear Regression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred1 = regressor.predict(X_test)
accuracy = regressor.score(X_test,Y_test)
print('Accuracy of the model is',accuracy*100,'%')
#print(Y_pred1)

#2. Random Forest (Decision Tree(s)) Regression model
x = 500
regressor = RandomForestRegressor(n_estimators = x)
regressor.fit(X,Y)
accuracy = regressor.score(X_train,Y_train)
Y_pred = regressor.predict(X_test)
print('Accuracy of the model with',str(x),'n_estimators','=',accuracy*100,'%')
#The most accurate result is getting predicted by Random Forest regression,
#so we are going to save the
#results in a csv file
with open('predictedscores_bythebest_algorithm_stub.csv', 'w', ) as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow([ 'Name', 'Predicted IMDB score(LR)', 'Predicted IMDB score(RF)' , 'Actual IMDB score'])
    wr.writerow(['' ,'' ,'' ,'' , 'Mean absolute error: ' + str(metrics.mean_absolute_error(Y_test, Y_pred))])
    wr.writerow([ '','' , '','' , 'Mean error: ' + str(metrics.mean_squared_error(Y_test, Y_pred))])
    wr.writerow([ '', '', '', '', 'Mean squared error: ' + str(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))])
    for dul in zip(df['name'] ,Y_pred1, Y_pred):
        actual_score = df.loc[df['name'] == dul[0], 'rating'].values[0]
        wr.writerow([dul[0], dul[1], dul[2], actual_score]) #Saving results in another csv file

ax = plt.axes()
ax.scatter(Y_pred, Y_test)
plt.title('Best Fit Model')
ax.plot(np.linspace(5, 10), np.linspace(5,10))
plt.xlabel('Predicted IMDb Rating')
plt.ylabel('Actual IMDb Rating')
plt.show()#Shows the best fit model in the form of a scatter plot

#saving the best fit model for later use
with open('model.pkl', 'wb') as pl:
    pickle.dump(regressor, pl)
