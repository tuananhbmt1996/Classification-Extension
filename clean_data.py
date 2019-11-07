import json
import nltk
import re
import csv
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pandas as pd
from tqdm import tqdm


def read_plot_from_corpus(file_link):
    plots = []
    with open(file_link, 'r',encoding="utf8") as f:
        reader = csv.reader(f, dialect='excel-tab')
        for row in tqdm(reader):
            plots.append(row)
    movie_description = []
    movie_id = []
    for row in plots:
        movie_description.append(row[1])
        movie_id.append(row[0])
    movies = pd.DataFrame({'movie_id': movie_id, 'plot': movie_description})
    return movies


def merge_data(movies,data):
    data['movie_id'] = data['movie_id'].astype(str)
    movies = pd.merge(movies, data[['movie_id', 'movie_name', 'genre']], on = 'movie_id')
    return movies


def convert_genres(movies):
    genre_new = []
    for genre in movies["genre"]:
        genre_new.append(list(json.loads(genre).values()))
    return genre_new


def remove_empty_rows(movies):
    movies_new = movies[~(movies['genre_new'].str.len() == 0)]
    return movies_new


def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]"," ",text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()
    return text


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def get_total_genres(movies):
    # an empty list
    genres = []
    # extract genres
    for i in movies['genre']:
      genres.append(list(json.loads(i).values()))
    # add to 'movies' dataframe
    movies['genre_new'] = genres
    all_genres = sum(genres,[])
    return len(set(all_genres))


def get_average_labels(data):
    count = 0
    for i in data['genre_new']:
        count +=  len(i)
    return count;
