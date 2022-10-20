
import pandas as pd
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
                        ,
                        index = ['a','b','c','d','e','f','g']
                        )

print(airports)
print(airports['Name']['a'])
print("yay")