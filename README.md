# Fake-News-Detection

# Web scraping:
We first collect data by web scraping 3 newspaper sites WSJ, NYTimes, and TampaBay for legitimate news content. In order to collect fake news data we used an API and gathered fake news content. The data is collected in a standard format of id, title, publication, author, published date, category, URL, content.

# Data merging and Data Preprocessing:
All the data collected is merged using pandas to form a combined dataset under Final_Data.csv. At the time of merging we label the data as “Real” and “Fake”. Data is cleaned by removing all the null values, special characters, Unicode characters which are identified by performing exploratory data analysis on the merged data. This cleaned data is stored under a separate CSV named CleanData.

# Data Cleansing:
· Removed characters from the content
· Updated empty rows with values
· Dropped Unwanted rows
· Removed stop words from content
· Punctuation has been handled

# Exploratory Data Analysis:
Exploratory data analysis usually performed to get an overview of the dataset gave us some insights about the data we had after merging. We realized a lot of data needs to be processed as shown in the below snapshot to generate a clean dataset. 

# Generating models:
The approach to implementing our idea is pretty simple, our ultimate goal is to classify a given piece of information. To achieve this we tried 5 models for classification and evaluated them to get the best one
Multinomial Naive Bayes
Random Forest
Support Vector Machine
Logistic Regression
XGBoost

# Pipeline Design:
Our code does the Data Merging for all the newspapers csv’s that we have and the MergeAllDataSingleFile() depends on four tasks for its input and to merge all of them into a single csv file. Later on the DataCleaning() depends on MergeAllDataSingle() and cleans the csv that is provided by it and stores into a csv. This acts as an input to our Model() where the metrics are calculated providing the accuracy of the model used for classification