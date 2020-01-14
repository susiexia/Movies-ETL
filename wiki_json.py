# %%
import json
import pandas as pd 
import numpy as np 

# %%
# EXTRACT process
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
