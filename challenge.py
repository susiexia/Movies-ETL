# %%
import json
import pandas as pd 
import numpy as np 
import re
from config import db_password
from sqlalchemy import create_engine
import psycopg2 

import time
# %% 
'''
 ----------EXTRACT process-------------------------
 Assumption 1: upcoming data resources are as same formats
 Assumption 2: wiki_data has same alternate title
 Assumption 3: 'Box_office' and 'Budget' columns have consistent data type and format followed by assumed dollar-like Regex
 Assumption 4: 'release date' column have consistent datetime type and format followed by assumed regex rules
'''
# %%

file_dir = '/Users/susiexia/desktop/module_8/Movies-ETL/raw_data'


# %% [markdown]
# ----------TRANSFORM process-------------------------


# %%
# Create a transform function that pass 3 data resources:
def ETL_data(wiki_data, kaggle_metadata, movielens_ratings):
    # -------------------------------------------------------
    # extract kaggle and movielens
    # applied Assumption 1
    try:
        kaggle_metadata_df = pd.read_csv(f'{file_dir}/{kaggle_metadata}', low_memory=False)
        movielens_ratings_df = pd.read_csv(f'{file_dir}/{movielens_ratings}')
        # extract wiki_data
        with open(f'{file_dir}/{wiki_data}', 'r') as js_file:
            Wiki_raw = json.load(js_file)
    except:
        print('Unable extract the raw dataset')
    # filter wiki datasets by only contains director and Imdb information
    wiki_movies = [movie for movie in Wiki_raw 
                if ('Directed by' in movie or 'Director' in movie) 
                and ('imdb_link' in movie)
                and ('No. of episodes' not in movie) ]
    # -------------------------------------------------------
    # TRANSFORM wiki_data 
    # create alt_title cleaning FUNCTION and a inner fuction of merging specific columns
    # applied to assumption 2
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

    # CALL clean_movie funtion
    try:
        clean_movies_wiki = [clean_movie(movie) for movie in wiki_movies]
        # transfter into DataFrame
        wiki_movies_df = pd.DataFrame(clean_movies_wiki)
    except:
        print ('Unable transform wiki_data into pandas.Dataframe')
    # CLEAN ROWS in DF form
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    
    # remove columns which have over 90% null rows
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns.tolist() if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    # Regex instance
    # applied to Assumption 3
    # 'Budget' and 'Box_office' data format
    dollar_form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    dollar_form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illi?on)'

    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    running_form_one =r'(\d+)\s*m'
    running_form_two = r'(\d+)\s*ho?u?r?s?\s*(\d*)'

    # applied Assumptions 3
    # parse and clean BOX_OFFICE, BUDGET, RELEASE_DATE and RUNNING TIME in wiki_data

  
    def parse_dollars(s):
        # step 1: check if str type
        if type(s) != str:
            return np.nan
        # step 2: check if match search pattern and return it (use raw string)
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
        
    # call FUNCTIONS 
    try:
        # parse box_office column in wiki_data
        box_office_Series = wiki_movies_df['Box office'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)\
                            .str.replace(r'\$.*[-—–](?![a-z])', '$', regex = True)
        wiki_movies_df['box_office'] = box_office_Series.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
        # parse budget column in wiki_data
        budget_Series = wiki_movies_df['Budget'].dropna().map(lambda x: ' '.join(x) if type(x) == list else x)\
                            .str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
        wiki_movies_df['budget'] = budget_Series.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

        # parse release_date column in wiki_data
        release_date_Series = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
        wiki_movies_df['release_date'] = pd.to_datetime(release_date_Series.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

        # parse running_time column in wiki_data
        running_time_Series = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
        running_time_extract = running_time_Series.str.extract(f'({running_form_one}|{running_form_two})', flags=re.IGNORECASE)\
                        .apply(lambda col: pd.to_numeric(col, errors= 'coerce')).fillna(0)
        # convert hour to minute and make a Series then put it back into DF
        wiki_movies_df['running_time'] = running_time_Series.apply(lambda row: row[0]*60 +row[1] if row[2] == 0 else row[2], axis =1)
         
    except:
        wiki_movies_df['box_office'] = np.nan
        wiki_movies_df['budget'] = np.nan        
        wiki_movies_df['release_date'] = np.nan
        wiki_movies_df['running_time']

        print ('Wrong date format in wiki_data, unable parse this column')
        

    # drop previous columns
    wiki_movies_df.drop('Box office', axis=1, inplace=True)
    wiki_movies_df.drop('Budget', axis = 1, inplace = True)
    wiki_movies_df.drop('Release date', axis = 1, inplace = True)
    wiki_movies_df.drop('Running time', axis = 1, inplace = True)
   
    # return wiki_movies_df
    return wiki_movies_df


# %%
# %%
# -----------TRANSFORM process-------------------------

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
# ---------------------ACTION -------DEF FUNCTION---CLEAN MERGED DATAFRAME----------
# make a function that fills in missing data for a column pair and then drops the redundant column
def fill_missing_kaggle_data(df, kaggle_col, wiki_col):
    # check whether kaggle row has non-zero value
    df[kaggle_col] = df[kaggle_col].apply(lambda row: row[wiki_col] if row[kaggle_col] == 0 else row[kaggle_col], axis=1)
    # drop wiki redundent column
    df.drop(column = wiki_col, inplace = True)

# %%
# ---------------------ACTION -------CLEAN MERGED DATAFRAME----------
# drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
movies_df.drop(['title_wiki','release_date_wiki','Language','Production company(s)'], axis=1, inplace=True)

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

movies_df.to_sql(name='movies', con=engine)

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
