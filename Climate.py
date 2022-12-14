import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_data(file, year, countries):
    """
    Function to get input file and the specific countries and the years

    input: 
    filename (string)
    year (array of years)
    countries (array of countries)

    output:
    result (array of the required values for specific years)
    """
    df = pd.read_csv (file ,on_bad_lines='skip')
    result = []
    for k in range(0, len(year)):
        for j in range(0, len(countries)): 
            for i in range (0, len(df[ 'Country Name'])) :
                if df['Country Name'][i] == countries[j]:
                    result.append(df[year[k]][i])
    return result

def extract_values(result):
    """
    Function to separate the datas for each country

    input: 
    result (array of the required values for specific years)
    
    output:
    arr (array of the data for sepaarte countries)
    """
    n = len(countries)
    arr = []
    for i in range(0, len(result), n):
        arr.append((result[i:i + n]))
    return arr

##### Calculating Total Greenhouse Gas Emission (kt of CO2 equivalent) #####
file = "API_EN.ATM.GHGT.KT.CE_DS2_en_CsV_V2_4748515.csv"
year = ["1995", "2000", "2005", "2010", "2015"]
countries = ["Austria", "Belgium", "Croatia", "Denmark", "France", "Germany", "Netherlands", "Portugal", "Spain"]
result = extract_data(file, year, countries)            #passing the file to functioin to get the data
plot_arr = extract_values(result)                       #passing the result data to obtain values for individual countries
fig = plt.subplots(figsize = (12, 8))
barWidth = 0.10
X = np.arange(9)
plt.bar(X + 0.00, plot_arr[0], color = 'blue', width = 0.10,label = year[0])
plt.bar(X + 0.10, plot_arr[1], color = 'red', width = 0.10,label = year[1])
plt.bar(X + 0.20, plot_arr[2], color = 'green', width = 0.10,label = year[2])
plt.bar(X + 0.30, plot_arr[3], color = 'yellow', width = 0.10,label = year[3])
plt.bar(X + 0.40, plot_arr[4], color = 'pink', width = 0.10,label = year[4])
plt.bar(X + 0.50, plot_arr[5], color = 'cyan', width = 0.10,label = year[5])
plt.xlabel("European Countries", fontweight = "bold")
plt.title("Total Greenhouse Gas Emission (kt of CO2 equivalent)")
plt.xticks([r + barWidth for r in range(len(countries))], countries)
plt.legend()
plt.show()

##### Calculating GDP per Capita (current US$)) #####
file = "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4749595.csv"
year = ["1995", "2000", "2005", "2010", "2015", "2020"]
countries = ["Austria", "Belgium", "Croatia", "Denmark", "France", "Germany", "Netherlands", "Portugal", "Spain"]
result = extract_data(file, year, countries)            #passing the file to functioin to get the data
plot_arr = extract_values(result)                       #passing the result data to obtain values for individual countries
fig = plt.subplots(figsize = (12, 8))
barWidth = 0.10
X = np.arange(9)
plt.bar(X + 0.00, plot_arr[0], color = 'black', width = 0.10,label = year[0])
plt.bar(X + 0.10, plot_arr[1], color = 'violet', width = 0.10,label = year[1])
plt.bar(X + 0.20, plot_arr[2], color = 'brown', width = 0.10,label = year[2])
plt.bar(X + 0.30, plot_arr[3], color = 'orange', width = 0.10,label = year[3])
plt.bar(X + 0.40, plot_arr[4], color = 'olive', width = 0.10,label = year[4])
plt.bar(X + 0.50, plot_arr[5], color = 'maroon', width = 0.10,label = year[5])
plt.xlabel("European Countries", fontweight = "bold")
plt.title("GDP per Capita (current US$))")
plt.xticks([r + barWidth for r in range(len(countries))], countries)
plt.legend()
plt.show()

##### Arable Land #####
file = "API_AG.LND.ARBL.ZS_DS2_en_csv_v2_4749667.csv"
year = ["1995", "2000", "2005", "2010", "2015", "2020"]
countries = ["Austria", "Belgium", "Croatia", "Denmark", "France", "Germany", "Netherlands", "Portugal", "Spain"]
result = extract_data(file, year, countries)            #passing the file to functioin to get the data
plot_arr = extract_values(result)                       #passing the result data to obtain values for individual countries
fig , ax = plt.subplots(figsize = (12, 8))
ax.plot(plot_arr)
plt.xticks([r for r in range(len(year))], year)
plt.title("Arable Land")
plt.legend(countries)
plt.show()

##### Forest Area #####
file = "API_AG.LND.FRST.ZS_DS2_en_csv_v2_4748391.csv"
year = ["1995", "2000", "2005", "2010", "2015", "2020"]
countries = ["Austria", "Belgium", "Croatia", "Denmark", "France", "Germany", "Netherlands", "Portugal", "Spain"]
result = extract_data(file, year, countries)            #passing the file to functioin to get the data
plot_arr = extract_values(result)                       #passing the result data to obtain values for individual countries
fig , ax = plt.subplots(figsize = (12, 8))
ax.plot(plot_arr)
plt.xticks([r for r in range(len(year))], year)
plt.title("GDP per Capita (current US$))")
plt.legend(countries)
plt.show()

##### Tabulating the rural population #####
file = "API_1_DS2_en_csv_v2_4550063.csv"
year = ["1995", "2000", "2005", "2010", "2015", "2020"]
countries = ["Austria", "Belgium", "Croatia", "Denmark", "France", "Germany", "Netherlands", "Portugal", "Spain"]
result = extract_data(file, year, countries)            #passing the file to functioin to get the data
plot_arr = extract_values(result)                       #passing the result data to obtain values for individual countries
data = {year[0]:plot_arr[0], year[1]:plot_arr[1], year[2]:plot_arr[2], year[3]:plot_arr[3]}
table_data = pd.DataFrame(data, index=countries)
print(table_data)