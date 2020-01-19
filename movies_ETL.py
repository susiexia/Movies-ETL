# %%
import json
import pandas as pd 
import numpy as np 
import re
# pd.to_sql() need connection
from config import db_password
from sqlalchemy import create_engine
import psycopg2 # postgres adaptor 


import time
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
#  # ----------TRANSFORM PART 2(preprocess): ----convert list to string--------BOX_OFFICE CLEAN ------------
# lambda + map() to pick up 135 not_a_string rows in box_office Series
# check the amount

#box_office_Series[box_office_Series.map(lambda x: type(x) != str)]

# transform list type into string by 'a space'.join(), apply() and lambda function 
box_office_Series = box_office_Series.apply(lambda x: ' '.join(x) if type(x) == list else x)

# box_office replace dash in the range type instance then use Series.tri.replace(regex style)
box_office_Series.str.replace(r'\$.*[-—–](?![a-z])', '$', regex = True)

# %%
#  # ----------TRANSFORM PART 2(preprocess): ----REGEX TEST--------BOX_OFFICE CLEAN ------------
                         
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
        s = re.sub(r'\$|\s|[a-zA-Z]', '', s)
        # float it and return 
        value = float(s)*10**6   # will be a float point number
        return value
    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
        s = re.sub(r'\$|\s|[a-zA-Z]','',s)
        value = float(s)*10**9
        return value
    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
        # remove dollar sign and commas
        s = re.sub(r'\$|,', '', s)
        value = float(s)
        return value
    else:
        return np.nan

# %%
#  # ----------TRANSFORM PART 2:  call function----BOX_OFFICE CLEAN ------------
# use str.extract() funtion to add a new column on DF, and apply function
# use [0] to extract the first column of  box_office_Series(it's a DF due to extract funtion)
wiki_movies_df['box_office'] = box_office_Series.str.extract(f'({form_one}|{form_two})', \
                                flags=re.IGNORECASE)[0].apply(parse_dollars)

# drop previouse column: box office
wiki_movies_df.drop('Box office', axis=1, inplace=True)

wiki_movies_df.head()
                            

# %%
#  # ----------TRANSFORM PART 3: (preprocess) BUDGET CLEAN ------------
# preprocess 1: drop nan rows
budget_Series = wiki_movies_df['Budget'].dropna()
# preprocess 2: convert list to str
budget_Series = budget_Series.map(lambda x: ' '.join(x) if type(x) == list else x)

# preprocess 3: convert range numbers
budget_Series = budget_Series.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

# check improper form

matches_form_one_bool = budget_Series.str.contains(form_one, flags = re.IGNORECASE)
matches_form_one_bool = budget_Series.str.contains(form_one, flags = re.IGNORECASE)

budget_Series[~matches_form_one_bool & ~matches_form_two_bool]
# %%
#  # ----------TRANSFORM PART 3: call function BUDGET CLEAN ------------

wiki_movies_df['budget'] = budget_Series.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

wiki_movies_df.drop('Budget', axis = 1, inplace = True)

# %%
#  # ----------TRANSFORM PART 4: (preprocess) RELEASE DATE CLEAN ------------
# preprocess : drop nan rows and convert list to str
release_date_Series = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

# preprocess 3: convert range numbers
# release_date_Series.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

# check improper form
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'
#  # ----------TRANSFORM PART 4: RELEASE DATE CLEAN ------------
# extract date as a column and use pandas build-in function to convert datetime format
try: 
    wiki_movies_df['release_date'] = pd.to_datetime(release_date_Series.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
except:
    wiki_movies_df['release_date'] = np.nan
    print('Wrong date format')
wiki_movies_df.drop('Release date', axis = 1, inplace = True)

# %%
#  # ----------TRANSFORM PART 5: (preprocess) Running Time CLEAN ------------

running_time_Series = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
len(running_time_Series)
# regular pattern (applied caputure group)
running_form_regular =r'(\d+)\s*m'
# hour + minute patterns (applied caputure group)
running_form_two = r'(\d+)\s*ho?u?r?s?\s*(\d*)'

# extract running time and use pd built_in funtion to_numeric convert str to number

running_time_extract_df = running_time_Series.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

# apply to_numeric and fillna() 
running_time_extract = running_time_extract_df.apply(lambda col: pd.to_numeric(col, errors= 'coerce')).fillna(0)

# convert hour to minute and make a Series then put it back into DF
# access each row by 'lambda row' function
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 +row[1] if row[2] == 0 else row[2], axis =1)
# drop origin one
wiki_movies_df.drop('Running time', axis = 1, inplace = True)

wiki_movies_df.head()
# %%
# --------KAGGLE ---TRANSFORM 1 ---adult column  ------------------
kaggle_metadata_df.dtypes

# check if 'adult' and 'video' are ready to convert to boolean
kaggle_metadata_df['adult'].value_counts()  # several bad data in 'adult'
# check where is the bad data other than True or False
corrupt_adult = kaggle_metadata_df['adult'].isin(['True', 'False'])
kaggle_metadata_df.loc[~corrupt_adult] 

# only keep rows where adult is False,(filter the whole df) and then drop the “adult” column.
# reduced kaggle_metadata DF by 12 rows and 1 columns
kaggle_metadata = kaggle_metadata_df[kaggle_metadata_df['adult']== 'False'].drop('adult', axis = 'columns')

kaggle_metadata.head()
# %%
# --------KAGGLE ---TRANSFORM 2 video column------------------
# convert data type from str to boolean

# create boolean column
# kaggle_metadata['video'] == 'True'
# assign back to DF
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
kaggle_metadata.video.dtypes
# %%
# --------KAGGLE ---TRANSFORM 3 budget, id, popularity and release_date column------------------
# convert those 3 columns from str type to numeric
# use astype for 'budget' column
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
# use pd.to_numeric for 'id' and 'popularity' columns
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'])
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors= 'raise')
# use to_datetime for 'release_date' column
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'],errors='raise')

# %%
# --------Ratings csv ---TRANSFORM ---------------------------
# check a summary description
ratings_df.info(null_counts= True)

# convert timstamp to pd.datetime data type, unit is second
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
ratings_df.info()

# %%
# --------Ratings csv ---TRANSFORM --stats info ---------------------------

# check data distribution by histogram 
ratings_df['rating'].plot(kind = 'hist')
# check data statistical information
ratings_df['rating'].describe()   #The median score is 3.5, the mean is 3.53

# %%
# ---------------TRANSFORM: MERGE DF (inner join)-------------------------------
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on ='imdb_id', suffixes=('_wiki','_kaggle'))
movies_df.head()
# %% [markdown]
# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle            Drop Wikipedia
# running_time             runtime                 Keep Kaggle; and fill in zeros with Wikipedia data
# budget_wiki              budget_kaggle           Keep Kaggle; and fill in zeros with Wikipedia data
# box_office               revenue                 Keep Kaggle; and fill in zeros with Wikipedia data
# release_date_wiki        release_date_kaggle     Drop Wikipedia
# Language                 original_language       Drop Wikipedia
# Production company(s)    production_companies    Drop Wikipedia

# %%
# ---------------------DECISION-----title column-----------------
# compare two title columns
movies_df[['title_wiki','title_kaggle']]
# confirm there is no any missing data or empty in kaggle title column
movies_df[(movies_df['title_kaggle'] == '')|(movies_df['title_kaggle'].isnull())]

# %%
# ---------------------DECISION-----runtime column-----------------
# draw a scatter plot to reveal any outlier and missing data
movies_df.fillna(0).plot(x='running_time', y='runtime', kind = 'scatter')
# %%
# ---------------------DECISION-----Budget column-----------------
# draw a scatter plot to reveal any outlier and missing data
movies_df.fillna(0).plot(x='budget_wiki', y='budget_kaggle', kind = 'scatter')


# %%
# ---------------------DECISION-----BOX OFFICE & Revenue column-----------------
# draw a scatter plot to reveal any outlier and missing data
movies_df.fillna(0).plot(x='box_office', y='revenue', kind = 'scatter')

# ---------------------DECISION----- narrow down scale BOX OFFICE & Revenue column-----------------
#  scatter plot for everything less than $1 billion in box_office(x axies)
narrow_down_box_movies_df = movies_df[movies_df['box_office']<10**9].fillna(0)
narrow_down_box_movies_df.plot(x='box_office', y='revenue', kind = 'scatter')

# %%
# ---------------------DECISION---- release date (datetime)column-----------------
movies_df[['release_date_wiki','release_date_kaggle']].plot(x= 'release_date_wiki',y='release_date_kaggle', style='.')

# %%
# investigate the outlier around 2006 (wiki)
# and decide to drop this invalid row
bad_data_index = movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index

movies_df.drop(bad_data_index)
# check missing data point in wiki (11 rows)
movies_df[movies_df['release_date_wiki'].isnull()]
# check missing data point in kaggle ( no missing data )
movies_df[movies_df['release_date_kaggle'].isnull()]
# %%
# ---------------------DECISION---- language column-----------------
# compare 2 language columns missing data amount
movies_df['Language'].apply(lambda x: \
                    tuple(x) if type(x) == list else x).value_counts(dropna=False)

movies_df['original_language'].value_counts(dropna=False)
# %%
# ---------------------DECISION---- language column-----------------
movies_df[['Production company(s)','production_companies']]


# %%
# ---------------------ACTION -------CLEAN MERGED DATAFRAME----------
# drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
movies_df.drop(['title_wiki','release_date_wiki','Language','Production company(s)'], axis=1, inplace=True)

# %%
# ---------------------ACTION -------DEF FUNCTION---CLEAN MERGED DATAFRAME----------
# make a function that fills in missing data for a column pair and then drops the redundant column


def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)

# %%
# ---------------------ACTION call funtion----CLEAN MERGED DATAFRAME------
# no any assigned variable, call clean funtion directly
fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

movies_df

# %%
# ----------------CLEAN MERGED DATAFRAME----------------
# check any columns have only one value
for col in movies_df.columns:
    lst_to_tuples_function = lambda x: tuple(x) if type(x) == list else x 
    value_counts = movies_df[col].apply(lst_to_tuples_function).value_counts(dropna=False)
    
    if len(value_counts) == 1:
        print(col)   # 'video' only have one value: False

# %%
# drop 'video' column
movies_df.drop('video', axis =1, inplace= True)
# %%
# reoder the columns
movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]

# rename the columns
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)


# %%
# -----------------RATINGS .csv Transform----------
# firstly groupby 'userID' and 'rating', then get the count of (each rating of each movie)
rating_counts = ratings_df.groupby(['movieId','rating'], as_index= False).count()\
                    .rename({'userId':'count'}, axis=1).pivot(index='movieId', columns='rating',values='count')

# rename every columns use list comprehension
rating_counts.columns = ['rating_' +str(col) for col in rating_counts.columns]

# %%
# -----------------RATINGS .csv MERGE into main DF----------
# left merge into main table: movies_df
movies_with_ratings_df = pd.merge(movies_df, rating_counts, how='left', left_on='kaggle_id',right_index=True)

# fill NaN with zero in all rating columns
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

# %%
# -----------------LOAD MOVIES_DF TO Postgres-----------------
db_string = f'postgres://postgres:{db_password}@127.0.0.1:5432/movie_data'
# use sqlalchemy.create_engine to prepare parameter of pd.to_sql()
engine = create_engine(db_string)

movies_df.to_sql(name='movies', con=engine, if_exists ='replace')

# %%
# -----------------LOAD rating csv to Postgres-------
rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')



# %%
