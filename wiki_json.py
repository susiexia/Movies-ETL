# %%
import json
import pandas as pd 
import numpy as np 
import re
# pd.to_sql() need connection
from sqlalchemy import create_engine
import psycopg2 # postgres adaptor 
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
wiki_movies_raw_df = pd.DataFrame(wiki_movies_raw)
wiki_movies_raw_df.head()
wiki_movies_raw_df.columns.tolist()

# %%
# filter wiki datasets by only contains director and Imdb information
# in original list of dictionary use list comprehension
wiki_movies = [movie for movie in wiki_movies_raw 
                if ('Directed by' in movie or 'Director' in movie) 
                and ('imdb_link' in movie)
                and ('No. of episodes' not in movie) ]
len(wiki_movies)  #7076 rows
# %%
# create a new, filtered DataFrame after list comprehension
wiki_movies_director_df = pd.DataFrame(wiki_movies)
len(wiki_movies_director_df.columns.tolist())
# %%
# test data 
#wiki_movies_raw_df.loc[wiki_movies_raw_df['Arabic'].notnull()]
#wiki_movies_raw_df['year'].dtypes
#wiki_movies_raw_df['Arabic'].value_counts()
# %%
# find alternate title columns
wiki_column_lst = wiki_movies_director_df.columns.tolist()
sorted(wiki_column_lst)
# %%
# test of detemining alternative title column
# wiki_foreign_movies = [movie for movie in wiki_movies_raw if 'Literally' in movie]
# wiki_foreign_movies[:1]
 # %%
 # ----------TRANSFORM PART 1-1:  CLEAN COLUMNS in python form------------
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
# inner function for merging column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
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

# %%
# CALL clean function in list comprehension
clean_movies_ListofDict = [clean_movie(movie) for movie in wiki_movies]
# transfter into DataFrame
wiki_movies_df = pd.DataFrame(clean_movies_ListofDict)
sorted(wiki_movies_df.columns.tolist())


# %%
 # ----------TRANSFORM PART 1-2:  CLEAN ROWS in DF form------------
# use Regex to EXTRACT imdb_ID
# import re is not nessesary because Series.str.extract() asking for regex in parenthesis
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
# use imdb_id as identifier to drop duplicates rows
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()

# %%
 # ----------TRANSFORM PART 1-3:  CLEAN mostly null COLUMNS in DF form BY using list comprehension ------------
# check every column's null count
# remove columns which have over 90% null rows # now 21 columns

wiki_columns_to_keep = [column for column in wiki_movies_df.columns.tolist() if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
# alter DF on selected columns 
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
wiki_movies_df.head()
#wiki_movies_df.dtypes
# -----191 columns reduced to 21 columns and 7033 rows now
# %%

#  # ----------TRANSFORM PART 2:  BOX_OFFICE CLEAN ------------
# drop null rows of box office (1548 null rows)(contains 5485 valid rows)
box_office_Series = wiki_movies_df['Box office'].dropna()
len(box_office_Series)
# %%
#  # ----------TRANSFORM PART 2(prework): ----convert list to string--------BOX_OFFICE CLEAN ------------
# lambda + map() to pick up 135 not_a_string rows in box_office Series
# check the amount

#box_office_Series[box_office_Series.map(lambda x: type(x) != str)]

# transform list type into string by 'a space'.join(), apply() and lambda function 
box_office_Series = box_office_Series.apply(lambda x: ' '.join(x) if type(x) == list else x)

# box_office replace dash in the range type instance then use Series.tri.replace(regex style)
box_office_Series.str.replace(r'\$.*[-—–](?![a-z])', '$', regex = True)

# %%
#  # ----------TRANSFORM PART 2(prework): ----REGEX TEST--------BOX_OFFICE CLEAN ------------
                         
# box_office form-1 : like $ 123.4 million” (or billion)
form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
# use pd.Series.str.contains(re) to determine whether contains form_one and sum() 
matches_form_one_bool = box_office_Series.str.contains(form_one, flags = re.IGNORECASE)

# box_office form-1 : like $123,456,789 or $ 1.234
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illi?on)'
matches_form_two_bool = box_office_Series.str.contains(form_two, flags = re.IGNORECASE)

# element-wise logical operators (&, ~, |) to check remaining wrong format
remaining_not_clean = box_office_Series[~matches_form_one_bool & ~matches_form_two_bool]

# %%
#  # ----------TRANSFORM PART 2:  BOX_OFFICE CLEAN  build a funtion based on prework------------
def parse_dollars(s):
    # step 1: check if str type
    if type(s) != str:
        return np.nan
    # step 2: check if match search pattern and return it (use raw string)
    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
        # remove and replace $ and space and 'million' word
        s = re.sub('\$|\s|[a-zA-Z]', '', s)
        # float it and return 
        value = float(s)*10**6
        return value
    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
        s = re.sub('\$|\s|[a-zA-Z]','',s)
        value = float(s)*10**9
        return value
    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
        # remove dollar sign and commas
        s = re.sub('\$|,', '', s)
        value = float(s)
        return value
    else:
        return np.nan

# %%
#  # ----------TRANSFORM PART 2:  call function----BOX_OFFICE CLEAN ------------
# use str.extract() funtion to add a new column on DF, and apply function
wiki_movies_df['box_office'] = box_office_Series.str.extract(f'({form_one}|{form_two})', \
    flags=re.IGNORECASE)[0].apply(parse_dollars)

# drop previouse column: box office
wiki_movies_df.drop('Box office', axis=1, inplace=True)

wiki_movies_df.head()
                            

# %%
