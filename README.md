# Book Recommendation System Machine Learning -Unsupervised learning
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings; warnings.simplefilter('ignore')
from scipy.stats import pearsonr

#  Dataset Loading
from google.colab import drive
drive.mount('/content/drive')


# Load Dataset
import pandas as pd
books = pd.read_csv('/content/drive/MyDrive/Copy of Books.csv')
users = pd.read_csv('/content/drive/MyDrive/Copy of Users.csv')
ratings = pd.read_csv('/content/drive/MyDrive/Copy of Ratings.csv')

# Dataset First View
# Dataset First Look
print('books data frame')
books.head(5)
print('users data frame')
users.head(5)

print('ratings data frame')
ratings.head(5)

# Dataset Rows & Columns count

print('books dataset shape-', books.shape)

print('users dataset shape-', users.shape)

print('ratings dataset shape-', ratings.shape)

# Dataset Information
books.info()
users.info()
ratings.info()

# Duplicate Values
# duplicate value count
print('books dataset duplicate values-',books.duplicated().sum())

print('users dataset duplicate values-',users.duplicated().sum())

print('ratings dataset duplicate values-',ratings.duplicated().sum())

# Missing Values/Null Values
# Missing Values/Null Values Count
print('books dataset null values-',books.isnull().sum())

print('users dataset null values-',users.isnull().sum())
print('ratings dataset null values-',ratings.isnull().sum())

# Visualizing the missing values, since age in users df has most missing values visualizing by pie chart
# Visualizing the missing values, since age in users df has most missing values visualizing by pie chart

# Calculate the number of missing values in the 'Age' column
missing_values_count = users['Age'].isnull().sum()

# Calculate the number of non-missing values in the 'Age' column
total_values_count = len(users['Age']) - missing_values_count

# Creating a pie chart to visualize missing values
labels = ['Missing Values', 'Non-Missing Values']
sizes = [missing_values_count, total_values_count]
colors = ['red', 'green']
explode = (0.1, 0)  # explode the 1st slice (Missing Values)
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart of Missing Values from users dataframe 'Age' Column")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#  Understanding Your Variables
# Data info
# Dataset Columns
books.columns
ratings.columns
users.columns
# Dataset Describe
books.describe()
users.describe()
ratings.describe()

# Variables Description
# Check Unique Values for each variable.
#Unique Values for each Authors.
books['Book-Author'].unique()
#Unique Values for each publications.
books['Publisher'].unique()

# Data Wrangling
books.isnull().sum()
# removing null values from books dataset book-author and publisher
books.dropna(subset= ['Book-Author','Publisher'], inplace = True)

#  Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables

# Chart - 1 visualization code
plt.figure(figsize=(10,6))
ax=sns.countplot(y='Book-Author',palette = 'dark', data = books, order = books['Book-Author'].value_counts().index[0:10])
ax.set_xlabel('Count', fontsize=14)
ax.set_ylabel('Book Author', fontsize=14)
plt.title("Top 10 Author's written most number of books")
plt.show()

# chart - 2 Top 10 publishers with most books published
# Chart - 2 visualization code
plt.figure(figsize=(10,6))
ax=sns.countplot(y='Publisher', palette = 'deep',data = books, order = books['Publisher'].value_counts().index[0:10])
ax.set_xlabel('Count', fontsize=14)
ax.set_ylabel('Publisher', fontsize=14)
plt.title('Top 10 publishers with most books published')
plt.show()

# Chart - 3 Top 20 Years with highest books published
# Chart -  visualization code
plt.figure(figsize=(10,6))
ax=sns.countplot(y='Year-Of-Publication', palette = 'deep',data = books, order = books['Year-Of-Publication'].value_counts().index[0:20])
ax.set_xlabel('Count', fontsize=14)
ax.set_ylabel('Year-Of-Publication', fontsize=14)
plt.title('Top 20 year with highest books published')
plt.show()

# Chart - 4 Top 10 books with highest count of ratings given by users
#merging books and ratings table
books_ratings = pd.merge(ratings,books, on = 'ISBN')
books_ratings.shape

#books_with_ratings table with ratings given by users
books_with_ratings = books_ratings[books_ratings['Book-Rating'] >0]
books_with_ratings.shape

# Chart - 4 visualization code
plt.figure(figsize=(10,6))
ax=sns.countplot(y='Book-Title', palette = 'deep',data = books_with_ratings, order = books_with_ratings['Book-Title'].value_counts().index[0:10])
ax.set_xlabel('Count', fontsize=14)
ax.set_ylabel('Book-Title', fontsize=14)
plt.title('Top 10 books with highest count of ratings given by users')

# Chart - 5 Ratings value counts
# creating a table with 'Book-ratings' with value counts i,e what rating was given to most of books by users
ratings_value_counts = books_with_ratings['Book-Rating'].value_counts().reset_index()

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
ax = sns.barplot(x=ratings_value_counts.index, y=ratings_value_counts['Book-Rating'], palette='muted')
ax.set_xlabel('Book Ratings', fontsize=14)
ax.set_ylabel('Count of Ratings', fontsize=14)
plt.title('Ratings Value Counts')
plt.show()

# 5. Data Cleaning and Data Manipulation
# 1. Feature Manipulation
# merging books and ratings table
books_ratings = pd.merge(ratings,books, on = 'ISBN')
print("shape of 'books ratings table' including books being rated and not rated by users-",books_ratings.shape)

#books_with_ratings table with ratings given by users
books_with_ratings = books_ratings[books_ratings['Book-Rating'] >0]
print("shape of 'books with ratings table' including only books that been rated by users-",books_with_ratings.shape)

# 2. Feature Selection
books_with_ratings.columns

# 6. ML Model Implementation
# 1. User similarity by ratings
# 1.1 From 'books_with_ratings' table lets check how many users have given rating to books
'''from books_with_ratings grouping by user_id, aggregating count of Book-Rating
gives how many books were rated by each user
----------books_with_ratings = books_ratings[books_ratings['Book-Rating'] >0]  '''

user_ratings_count = books_with_ratings.groupby('User-ID').count()['Book-Rating'].reset_index()
user_ratings_count.rename(columns={'Book-Rating':'number_of_books_rated'},inplace = True)

# shape of user_ratings_count
print("size of table for 'user rated atleast 1 book'-",user_ratings_count.shape)
print("size of table for 'book rated by atleast by 1 user'-",books_with_ratings.shape)

# 1.2 Lets visualize a distribution plot where users are divided into bins ranging from 0-500 based on number of ratings given to books
plt.figure(figsize=(10, 6))
ax = sns.histplot(user_ratings_count['number_of_books_rated'], bins=[0, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500], kde=False)
ax.set_xlabel('Number of Books Rated by each user', fontsize=14)
ax.set_ylabel('Count of Users rated books', fontsize=14)
plt.title("Count of Users Rated Books", fontsize=16)
plt.show()

# 1.3 To understand users based on range of books they have rated from 1-5,5-10 and so on
# Initialize lists to store counts for each range
counts = [0] * 8

# Defining the ranges
ranges = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 40), (41, 50), (51, float('inf'))]

# Loop through the number of books rated by each user
for i in user_ratings_count['number_of_books_rated']:
    for idx, (start, end) in enumerate(ranges):
        if start <= i <= end:
            counts[idx] += 1
            break

# Create a DataFrame to store the results
data = {'Range_of_users': [f'{start}-{end}' if end != float('inf') else f'{start}+' for start, end in ranges],
        'no_of_ratings_given': counts}

user_ratings_count_range = pd.DataFrame(data)
user_ratings_count_range

# reconfirming that users rated more than 50 books are 1150 users
user_ratings_count[user_ratings_count['number_of_books_rated']>50].shape

# 1.4 Considering users rated atleast 50 books for recommendations system-
# filerring table with users rated more than 50 books only
user_rated_greaterthan_50books =user_ratings_count[user_ratings_count['number_of_books_rated']>50]
user_rated_greaterthan_50books

# 2. Books similarity by ratings
# filtering books from 'books with rating' table with 'user_rated_greaterthan_50books
filtered_user_rated_greaterthan_50books = books_with_ratings[books_with_ratings['User-ID'].isin(user_rated_greaterthan_50books['User-ID'])]
print('size of data after filtering users who has given 50+ ratings',filtered_user_rated_greaterthan_50books.shape)

'''from 'filtered_user_rated_greaterthan_50books' grouping by 'Book-Title' aggregating count of books_Rating
gives how many ratings were given for each book '''
filtered_book_ratings_count = filtered_user_rated_greaterthan_50books.groupby('Book-Title').count()['Book-Rating'].reset_index()
filtered_book_ratings_count.rename(columns={'Book-Rating':'number_of_ratings_given_to_book'},inplace = True)
print('size of filtered_book_ratings_count-',filtered_book_ratings_count.shape)

# 2.2 Lets visualize distribution plot of books by count of ratings given
plt.figure(figsize=(10, 6))
ax = sns.histplot(filtered_book_ratings_count['number_of_ratings_given_to_book'], bins=[0, 5, 10, 20, 50, 100, 150], kde=False)
ax.set_xlabel('Number of Books', fontsize=14)
ax.set_ylabel('Count of rating', fontsize=14)
plt.title("Distribution of books by users rated", fontsize=16)
plt.show()

# 2.3 To understand books based on range of ratings given by users from 1-5,5-10 and so on
# Initialize lists to store counts for each range
counts = [0] * 8

# Defining the ranges
ranges = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 40), (41, 50), (51, float('inf'))]

# Loop through the number of ratings given for each book
for i in filtered_book_ratings_count['number_of_ratings_given_to_book']:
    for idx, (start, end) in enumerate(ranges):
        if start <= i <= end:
            counts[idx] += 1
            break

# Create a DataFrame to store the results
data1 = {'range_of_ratings': [f'{start}-{end}' if end != float('inf') else f'{start}+' for start, end in ranges],
        'no_of_books': counts}

book_ratings_count_range = pd.DataFrame(data1)
book_ratings_count_range

#reconfirming that book with 50+ number of ratings
filtered_book_ratings_count[filtered_book_ratings_count['number_of_ratings_given_to_book']>15].shape

# 2.4 Considering Books with 15+ number of ratings and 50+ rated users combination for recommendation system
Books_with_greaterthan_50ratings = filtered_book_ratings_count[filtered_book_ratings_count['number_of_ratings_given_to_book']>15]
Books_with_greaterthan_50ratings.head()

