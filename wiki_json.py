# %%
import json
import pandas as pd 
import numpy as np 

# %%
# load a json file into python, use f_string
file_dir = '/Users/susiexia/desktop/module_8/Movies-ETL'
# use WITH statement to open and read file
with open(f'{file_dir}/wikipedia.movies.json', 'r') as js_file:
    wiki_movies_raw = json.load(js_file)
len(wiki_movies_raw)
wiki_movies_raw[-5:]

# %%
