# %%
import json
import pandas as pd 
import numpy as np 

# %%
# ----------EXTRACT process-------------------------
# load Wiki source (json file) into python, use f_string
file_dir = '/Users/susiexia/desktop/module_8/Movies-ETL/raw_data'
# use WITH statement to open and read file
with open(f'{file_dir}/wikipedia.movies.json', 'r') as js_file:
    wiki_movies_raw = json.load(js_file)
len(wiki_movies_raw)
wiki_movies_raw[-5:]

# %%
# load MovieLens source(csv file) from Kaggle
kaggle_metadata_df = pd.read_csv(f'{file_dir}/movies_metadata.csv', low_memory=False)
ratings_df = pd.read_csv(f'{file_dir}/ratings.csv')
#kaggle_metadata_df.head()
#ratings_df.head()
kaggle_metadata_df.sample(n=5)

kaggle_metadata_df.tail()

# %%
# -----------TRANSFORM process-------------------------
# wiki source tranform into DataFrame to inspect
wiki_movies_df = pd.DataFrame(wiki_movies_raw)
wiki_movies_df.head()
wiki_movies_df.columns.tolist()
#wiki_movies_df.index

# %%
# filter wiki datasets by only contains director and Imdb information
# in original list of dictionary use list comprehension
wiki_movies = [movie for movie in wiki_movies_raw 
                if ('Directed by' in movie or 'Director' in movie) 
                and ('imdb_link' in movie)
                and ('No. of episodes' not in movie) ]
len(wiki_movies)
# %%
# create a new, filtered DataFrame after list comprehension
wiki_movies_director_df = pd.DataFrame(wiki_movies)
len(wiki_movies_director_df.columns.tolist())
# %%
# test data 
wiki_movies_df.loc[wiki_movies_df['Arabic'].notnull()]
wiki_movies_df['year'].dtypes
wiki_movies_df['Arabic'].value_counts()
# %%
# find alternate title columns
wiki_column_lst = wiki_movies_director_df.columns.tolist()
sorted(wiki_column_lst)
# %%
# test of detemining alternative title column
wiki_foreign_movies = [movie for movie in wiki_movies_raw if 'Literally' in movie]
wiki_foreign_movies[:1]
 # %%
 # create alt_title cleaning FUNCTION and a inner fuction of merging specific columns
def clean_movie(movie):
    # Step 1: Make an empty dict to hold all of the alternative titles.
    
    movie = dict(movie)  # make a nondestructure copy
    alt_titles_dict = dict()
    # Step 2: Loop through a list of all alternative title keys.
    target_alt_titles = ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']
    for key in target_alt_titles:
    # Step 2a: Check if the current key exists in the movie object.
        if key in movie:
    # Step 2b: If so, remove the key-value pair and add to the alternative titles dictionary.
            alt_titles_dict[key] = movie[key] # move specific key-value pair into new dictionary
            movie.pop(key)  # remove orginal key-value pairs
    # Step 3: After looping through every key, add the alternative titles dict to the movie object.
    if len(alt_titles_dict) > 0:  # make sure it's valid, then add back to movie, like DF adding column method
        movie['alt_titles'] = alt_titles_dict
# inner function(call it ONLY in the outer FUNCTION)
    def change_column_name(old_name, new_name):
        movie[new_name] = movie.pop(old_name)
        # no return needed
# call inner function on every instance in outer function's scope
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')
# return of outer function
    return movie
    


    
