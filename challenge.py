# %%
import json
import pandas as pd 
import numpy as np 
import re
from config import db_password
from sqlalchemy import create_engine
import psycopg2 

import time
# %% [markdown]
# Build an automated ETL process

# %% [markdown]
## CREATE A FUNCTION through Etraxt-Transform-Load process

# %%
def ETL_data(wiki_data, kaggle_data, movielens_ratings):
    '''
    ----------EXTRACT process-------------------------
    Assumption 1:Input data resources are keeping same formats, same columns name, 
                same column's order and same data types.
    Assumption 2: wiki_data has same alternate titles
    Assumption 3: wiki_data: 'Box_office', 'Budget','release date' and 'running_time 'columns 
                have consistent data types and format followed by assumed Regex rules
    Assumption 4: kaggle_data: 'budget', 'id', 'popularity' and 'release_date' columns
                have consistent and appropriate data types, no any errors would be raised.
                when use astype(), to_numeric() and to_datetime() funtions.
    Assumption 5: The common column for merge dataframe would be unchanged. 
                Merge two dataframes, wiki_data and kaggle_data, woulbe be on 'imdb_id' column.
    Assumption 6: The movielens_ratings dataset would be able to groupby two columns 'movieId','rating'.
    Assumption 7: The PostgreSQL owner and server names keep unchanged. 
    '''
    # -------------------------------------------------------------------
    # applied Assumption 1
    file_dir = '/Users/susiexia/desktop/module_8/Movies-ETL/raw_data'

    try:
        kaggle_data_df = pd.read_csv(f'{file_dir}/{kaggle_data}', low_memory=False)
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
        if len(alt_titles_dict) > 0:  # make sure it's valid, then add back to movie
            movie['alt_titles'] = alt_titles_dict
    
    
        # Inner function for merging column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
                # no return needed
        # Call inner function on every instance in outer function's scope
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
    

    # parse box_office column in wiki_data
    box_office_Series = wiki_movies_df['Box office'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)\
                            .str.replace(r'\$.*[-—–](?![a-z])', '$', regex = True)
    wiki_movies_df['box_office'] = box_office_Series.str.extract(f'({dollar_form_one}|{dollar_form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    # parse budget column in wiki_data
    budget_Series = wiki_movies_df['Budget'].dropna().map(lambda x: ' '.join(x) if type(x) == list else x)\
                            .str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    wiki_movies_df['budget'] = budget_Series.str.extract(f'({dollar_form_one}|{dollar_form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    # parse release_date column in wiki_data
    release_date_Series = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    wiki_movies_df['release_date'] = pd.to_datetime(release_date_Series.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    # parse running_time column in wiki_data
    running_time_Series = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    running_time_extract_df = running_time_Series.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    running_time_extract = running_time_extract_df.apply(lambda col: pd.to_numeric(col, errors= 'coerce')).fillna(0)

    # convert hour to minute and make a Series then put it back into DF
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 +row[1] if row[2] == 0 else row[2], axis =1)

    
    wiki_movies_df['box_office'] = np.nan
    wiki_movies_df['budget'] = np.nan        
    wiki_movies_df['release_date'] = np.nan
    wiki_movies_df['running_time'] = np.nan        

    # drop previous columns
    wiki_movies_df.drop('Box office', axis=1, inplace=True)
    wiki_movies_df.drop('Budget', axis = 1, inplace = True)
    wiki_movies_df.drop('Release date', axis = 1, inplace = True)
    wiki_movies_df.drop('Running time', axis = 1, inplace = True)
   
    # return wiki_movies_df

    # -------------------------------------------------------
    # TRANSFORM kaggle_data
    
    # filter kaggle_data with only adult is False
    kaggle_metadata = kaggle_data_df[kaggle_data_df['adult']== 'False'].drop('adult', axis = 'columns')
    # convert 'video' data type from str to boolean
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    # applied to Assumption 4:
    # convert Kaggle's budget, id, popularity and release_date columns data type
    try:
        kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
        kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'])
        kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors= 'raise')
        kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'],errors='raise')
    except:
        print('inappropriate data types in Kaggle_megadata')
    
    
    # -------------------------------------------------------
    # TRANSFORM movielens_ratings_df
    # convert rating's timestamp into datetime
    movielens_ratings_df['timestamp'] = pd.to_datetime(movielens_ratings_df['timestamp'], unit='s')

    # -------------------------------------------------------
    # TRANSFORM MERGE wiki_movies_df with kaggle_metadata
    # Assumption 5
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on ='imdb_id', suffixes=('_wiki','_kaggle'))

    # PARSE MERGED dataframe
    # Assumption 6
    movies_df.drop(['title_wiki','release_date_wiki','Language','Production company(s)'], axis=1, inplace=True)

    # make a function that fills in missing data for a column pair and then drops the redundant column
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)
    
    #  CALL funtion----CLEAN MERGED DATAFRAME------
    # no any assigned variable, call clean funtion directly
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # check any columns have only one value
    for col in movies_df.columns:
        value_counts = movies_df[col].apply(lambda x: tuple(x) if type(x) == list else x )\
                        .value_counts(dropna=False)
    
        if len(value_counts) == 1:
            movies_df.drop(col, axis =1, inplace= True)
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
    
    # -------------------------------------------------------
    # TRANSFORM MERGE wiki_movies_df with rating
    
    
    # Assumption 6
    # firstly groupby 'userID' and 'rating', then get the count of (each rating of each movie)
    rating_counts = movielens_ratings_df.groupby(['movieId','rating'], as_index= False).count()\
                    .rename({'userId':'count'}, axis=1).pivot(index='movieId', columns='rating',values='count')
    
    # rename every columns use list comprehension
    rating_counts.columns = ['rating_' +str(col) for col in rating_counts.columns]

    # left merge into main table: movies_df
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, how='left', left_on='kaggle_id',right_index=True)

    # fill NaN with zero in all rating columns
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)




    # -------------------------------------------------------
    # LOAD to PostgreSQL
    # applied Assumption 7
    engine = create_engine(f'postgres://postgres:{db_password}@127.0.0.1:5432/movie_data')

    movies_with_ratings_df.to_sql(name='movies', con=engine, if_exists ='replace')

    print(f'''
        ETL process is successful done.\n Extracting from {wiki_data}, {kaggle_data}and {movielens_ratings}.\n 
        Results movies table in PostgreSQL.''')


# %% 
# -------------------------CALL function---------------------------
ETL_data("wikipedia.movies.json", "movies_metadata.csv", "ratings.csv")


# %%
