
#---------------------------- 3 ---------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
airports = pd.Series([
                      'Seattle-Tacoma', 
                      'Dulles', 
                      'London Heathrow', 
                      'Schiphol', 
                      'Changi', 
                      'Pearson', 
                      'Narita'
                      ])
print(airports,"\n", airports[2])


for value in airports:
    print(value) 

print("\n\n\n\n\n")

airports = pd.DataFrame([
                        ['Seatte-Tacoma', 'Seattle', 'USA'],
                        ['Dulles', 'Washington', 'USA'],
                        ['London Heathrow', 'London', 'United Kingdom'],
                        ['Schiphol', 'Amsterdam', 'Netherlands'],
                        ['Changi', 'Singapore', 'Singapore'],
                        ['Pearson', 'Toronto', 'Canada'],
                        ['Narita', 'Tokyo', 'Japan']
                        ])

print(airports,"\n",airports[0][2], "\n\n\n")
airports = pd.DataFrame([
                        ['Seatte-Tacoma', 'Seattle', 'USA'],
                        ['Dulles', 'Washington', 'USA'],
                        ['London Heathrow', 'London', 'United Kingdom'],
                        ['Schiphol', 'Amsterdam', 'Netherlands'],
                        ['Changi', 'Singapore', 'Singapore'],
                        ['Pearson', 'Toronto', 'Canada'],
                        ['Narita', 'Tokyo', 'Japan']
                        ],
                        columns = ['Name', 'City', 'Country']
                       
                        )

print(airports)
print(airports['Name'][1])
print("yay")

#---------------------  4 -----------------------------



print("4444444444444444444444444444444444444444444444444444\n",airports.head(3),"\n", airports.tail(3),"\n", airports.shape, "\n")

print("\n",airports.info())

#---------------------------- 5 ---------------------------------

print(airports['City'], "\n\n\n")

print(airports[['Name', 'Country']], "\n\n\n")

print(airports.iloc[0,1],"\n\n",airports.iloc[2,2],"\n\n",airports.iloc[:,:])

print(airports.iloc[0:2,:],"\n\n",airports.iloc[:,0:2],"\n\n", airports.iloc[:,[0,2]], "\n\n",

airports.loc[[0,2],['Name', 'Country']]

)

#---------------------------- 6 ---------------------------------
#Name,City,Country 
#Seattle-Tacoma,Seattle,USA
#Dulles,Washington,USA
#Heathrow,London,United Kingdom
#Schiphol,Amsterdam,Netherlands
#Changi,Singapore,Singapore
#Pearson,Toronto,Canada
#Narita,Tokyo,Japan
#---------------------------- 7 -----------------------------------

airports_df = pd.read_csv('Data/airports.csv.txt', error_bad_lines= False)
print(airports_df)

airports_af = pd.read_csv(
                          'Data/airportsNoHeaderRows.csv'
                           )
print("\n\n", airports_af)

airports_bf = pd.read_csv(
                          'Data/airportsNoHeaderRows.csv', header=None, names = ['Names', 'City', 'Country']
                           )


print("\n\n\n", airports_bf)

airports_df = pd.read_csv('Data/airportsBlankValues.csv.txt')
print(airports_df)

#airports_df.to_csv('Data/MyNewCSVFile.csv.txt')
airports_df.to_csv(
                   'Data/MyNewCSVFileNoIndex.csv.txt', 
                    index=False
                    )

#-------------------------------- 8 -----------------------------
delays_df = pd.read_csv('Data/flight_delays.csv.txt')

print("\n\n\n", delays_df.head())
print("\n\n\n",delays_df.keys())

new_df = delays_df.drop(columns=['ARR_TIME'])

print("\n\n\n", new_df.keys(), "\n\n", delays_df.head(), delays_df.keys())
delays_df.drop(columns=['ARR_TIME'], inplace=True)
print(delays_df.keys())

desc_df = delays_df.loc[:,['ORIGIN', 'DEST', 'OP_CARRIER_FL_NUM', 'OP_UNIQUE_CARRIER', 'TAIL_NUM']]
print(desc_df.head())

#----------------------------- 9 ------------------------------


delays_df = pd.read_csv('Data/Lots_of_flight_data.csv')
print(delays_df.head(), delays_df.keys(), "\n\n")
print(delays_df.info())

delay_no_nulls_df = delays_df.dropna()   # Delete the rows with missing values
print(delay_no_nulls_df.info())  

delays_df.dropna(inplace=True)
print(delays_df.info())

airports_df = pd.read_csv('Data/airportsDuplicateRows.csv.txt')

print(airports_df.head())
print(airports_df.duplicated())
airports_df.drop_duplicates(inplace=True)
print(airports_df)

#---------------------------- 10 -----------------------------

print(delays_df.shape, "\n\n")

X = delays_df.loc[:,['DISTANCE', 'CRS_ELAPSED_TIME']]
print(X.head(), "\n\n")

y = delays_df.loc[:,['ARR_DELAY']]
print(y.head(), "\n\n\n\n\n\n\n\n\n\n\n\n")

#X_train, X_test, y_train, y_test = train_test_split(
     #                                               X, 
     #                                               y, 
     #                                               test_size=0.3, 
     #                                               random_state=42
       #                                            )
#print(train_test_split(
       #                                             X, 
      #                                              y, 
      #                                              test_size=0.3, 
      #                                              random_state=42
     #                                              ))
     #
#print("\n\n\n",X_train.shape, X_test.shape,"\n\n", X_train.head(),"\n\n", y_train.shape, y_test.shape, "\n\n",y_train.head())

#-------------------- 11 --------------------------------
from sklearn.linear_model import LinearRegression

X = delays_df.loc[:,['DISTANCE', 'CRS_ELAPSED_TIME']]

# Move our labels into the y DataFrame
y = delays_df.loc[:,['ARR_DELAY']] 

# Split our data into test and training DataFrames
X_train, X_test, y_train, y_test = train_test_split( X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42
                                                   )           
regressor = LinearRegression()     # Create a scikit learn LinearRegression object
regressor.fit(X_train, y_train)    # Use the fit method to train the model using yo                                      y, 

#------------------------------ 12 --------------------------

y_pred = regressor.predict(X_test)
print(y_pred, "\n\n\n")


#--------------------------------- 13 -------------------------

from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R^2: ',metrics.r2_score(y_test, y_pred))

#---------------------------------- 14 --------------------------

print(type(y_pred), type(y_test))

airports_array = np.array(['Pearson','Changi','Narita'])
print("\n\n",airports_array)
print(airports_array[2], "\n\n")
airports_array2 = np.array([
  ['YYZ','Pearson'],
  ['SIN','Changi'],
  ['NRT','Narita']])
print(airports_array2)
print(airports_array2[0,0], "\n\n")

airports_zf = pd.DataFrame([['YYZ','Pearson'],['SIN','Changi'],['NRT','Narita']])
print(airports_zf)
print(airports_zf.iloc[0,1], "\n\n")

predicted_df = pd.DataFrame(y_pred)
print(predicted_df.head())

#------------------------ 15 -----------------------------



import matplotlib.pyplot as plt

#Check if there is a relationship between the distance of a flight and how late the flight arrives
delays_df.plot(
               kind='scatter',
               x='DISTANCE',
               y='ARR_DELAY',
               color='blue',
               alpha=0.3,
               title='Correlation of arrival and distance'
              )
plt.show()
delays_df.plot(
               kind='scatter',
               x='DEP_DELAY',
               y='ARR_DELAY',
               color='blue',
               alpha=0.3,
               title='Correlation of arrival and distance'
              )
plt.show()