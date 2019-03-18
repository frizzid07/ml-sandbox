import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Helper Functions
def title_from_index(index):
    return df[df.index == index]['title'].values[0]

def index_from_title(title):
    return df[df.title == title]['index'].values[0]

#Reading Training Data
df = pd.read_csv('movie_dataset.csv')
#print(df.columns)

#Select Relevant Features
features = ['keywords', 'cast', 'genres', 'director']

#Column of combined features
for x in features:
    df[x] = df[x].fillna('')
    
def combine_features(row):
    return row['keywords']+ ' ' +row['cast']+ ' ' +row['genres']+ ' ' +row['director']

df['combined_features'] = df.apply(combine_features, axis=1)
#print(df['combined_features'].head())

#Creating Count Matrix
cv = CountVectorizer()
ct_mat = cv.fit_transform(df['combined_features'])

#Compute Cosine Similarity
cos_sim = cosine_similarity(ct_mat)

#Input the movie liked by the user
movie_liked = input('Enter a Movie: ')

#Get index of liked movie
movie_index = index_from_title(movie_liked)
similar_movies = list(enumerate(cos_sim[movie_index]))

#Sort according to decreasing order of similarity
sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse=True)

#Print 10 similar movies
print('Similar Movies')
i = 0
for x in sorted_similar_movies:
    if title_from_index(x[0]) == movie_liked:
        continue
    else:
        print(title_from_index(x[0]))
    i += 1
    if i == 10:
        break