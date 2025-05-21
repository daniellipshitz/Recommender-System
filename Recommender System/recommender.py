# %% [markdown]
# ## Importing and linking databases

# %% [markdown]
# ## Movies

# %%
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import pgeocode
from sklearn.preprocessing import StandardScaler
import gensim.downloader as api
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from tensorflow.keras.layers import Embedding, LSTM, Dense, BatchNormalization, Input, Dropout, Concatenate, Flatten, Lambda, Add, Activation, GlobalAveragePooling1D, Dot, Multiply
from tensorflow.keras.models import Model, Sequential
from sklearn.decomposition import PCA
import umap

# %%
movies_lens_ratings = pd.read_csv("ml-100k/u1.base",sep='\t',header=None, names=["userId", "movieId", "rating", "timestamp"])

# %%
movies_lens_ratings.head()

# %%
movies_lens_items = pd.read_csv("ml-100k/u.item",
                    sep='|',
                    header=None,
                    encoding="latin1",
                    names=["id", "title", "date released", "video_release_date", "link", "unknown", 
    "Action", 
    "Adventure", 
    "Animation", 
    "Children's", 
    "Comedy", 
    "Crime", 
    "Documentary", 
    "Drama", 
    "Fantasy", 
    "Film-Noir", 
    "Horror", 
    "Musical", 
    "Mystery", 
    "Romance", 
    "Sci-Fi", 
    "Thriller", 
    "War", 
    "Western"])

# %%
movies_lens_items.head()

# %%
movies_lens_items['mod_year'] = movies_lens_items['title'].str.extract(r'\((\d{4})\)')[0].astype('Int64')
movies_lens_items['title'] = movies_lens_items['title'].str.extract(r'^(.*) \((\d{4})\)$')[0]
movies_lens_items['release_year'] = pd.to_datetime(movies_lens_items['date released'], format='%d-%b-%Y').dt.year.astype('Int64')
movies_lens_items = movies_lens_items.drop(['date released', 'video_release_date', 'link'], axis=1)

# %%
mismatched_years = movies_lens_items[movies_lens_items['mod_year'] != movies_lens_items['release_year']]

# %%
mismatched_years

# %%
#movies_lens_items = movies_lens_items.drop('mod_year', axis=1)

# %%
movies_lens_items.columns

# %%
#Find commas -> might need to fix title
commas = movies_lens_items[movies_lens_items['title'].str.contains(',', na=False)]
commas.head(75)

# %%
def fix_title(title):
    if not isinstance(title, str) or title.strip() == '':
        return title
    #translated titles
    title = re.sub(r'\s*\([^()]*\)$', '', title)
    
    # Revered titles
    title = re.sub(r'^(.*?),\s(The|A|An|Il|La|Le|Les|L\'|Der|Das)$', r'\2 \1', title)
    
    return title.strip()

movies_lens_items['title'] = movies_lens_items['title'].fillna('').astype(str).apply(fix_title)
movies_lens_items = movies_lens_items.drop_duplicates(subset=['title', 'release_year'])
movies_lens_items = movies_lens_items.drop_duplicates(subset=['id'])

# %%
movies_lens_items.shape

# %%
links = pd.read_csv("movie-data/links.csv", sep=',')
links = links.rename(columns={'movieId': 'id'})

# %%
links.head()

# %%
movie_data_items = pd.read_csv("movie-data/movies_metadata.csv", sep=',', encoding="latin1")

# %%
movie_data_items.head()

# %%
movie_data_items['release_year'] = pd.to_datetime(movie_data_items['release_date'], errors='coerce').dt.year.astype('Int64')
movie_data_items_title_mismatch = movie_data_items[movie_data_items['original_title'] != movie_data_items['title']]

# %%
movie_data_items_title_mismatch.head()

# %%
movie_data_items = movie_data_items.drop(['original_title', 'homepage', 'release_date', 'status', 'video', 'poster_path', 'adult', 'belongs_to_collection', 'release_date'], axis=1)

# %%
movies_lens_items

# %%
movie_data_items['imdb_id'] = movie_data_items['imdb_id'].str.replace('^tt', '', regex=True).astype('Int64')


# %%
movies_lens_items = pd.merge(
    movies_lens_items,
    links,
    on='id',
    how='left',
    suffixes=('', '_link')
)

# %%
movies_lens_items.head()

# %%
movie_data_merged = pd.merge(
    movie_data_items,
    links,
    on='id',
    how='left',
    suffixes=('', '_link')
)

# Find unmatched rows
unmatched_mask = movie_data_merged['imdbId'].isna()
unmatched_movies = movie_data_items[movie_data_items['id'].isin(movie_data_merged[unmatched_mask]['id'])]

# Try matching unmatched rows using imdb_id to imdbId
second_merge = pd.merge(
    unmatched_movies,
    links,
    left_on='imdb_id',
    right_on='imdbId',
    how='left',
    suffixes=('', '_link')
)

# Combine matched and newly matched rows
movie_data_final = pd.concat([
    movie_data_merged[~unmatched_mask],  # Successfully matched on id
    second_merge  # Matched on imdb_id or still unmatched
], ignore_index=True)

# Clean up duplicate columns
columns_to_drop = [col for col in movie_data_final.columns if col.endswith('_link')]
movie_data_items = movie_data_final.drop(columns=columns_to_drop)
movie_data_items = movie_data_items.rename(columns={'id': 'data_id'})

# %%
movie_data_items

# %%
# 1. Try merging on title+release_year
merged_step1 = pd.merge(
    movies_lens_items,
    movie_data_items,
    on=["title", "release_year"],
    how="left",
    suffixes=("", "_md")
)

# unmatched
mask_unmatched = merged_step1["imdb_id"].isna()
unmatched_df = movies_lens_items[movies_lens_items["id"].isin(merged_step1[mask_unmatched]["id"])]

# 2. merge title+mod_year instead
merged_step2 = pd.merge(
    unmatched_df,
    movie_data_items,
    left_on=["title", "mod_year"],
    right_on=["title", "release_year"],
    how="left",
    suffixes=("", "_md")
)

# still unmatched
mask_still_unmatched = merged_step2["imdb_id"].isna()
still_unmatched_df = unmatched_df[unmatched_df["id"].isin(merged_step2[mask_still_unmatched]["id"])]

# 3. title
merged_step3 = pd.merge(
    still_unmatched_df,
    movie_data_items,
    on="title",
    how="left",
    suffixes=("", "_md")
)

# Find rows that are still unmatched after third merge
mask_final_unmatched = merged_step3["imdb_id"].isna()
final_unmatched_df = still_unmatched_df[still_unmatched_df["id"].isin(merged_step3[mask_final_unmatched]["id"])]

# 4. merge imdbId
merged_step4 = pd.merge(
    final_unmatched_df,
    movie_data_items,
    left_on="imdbId",
    right_on="imdb_id",
    how="left",
    suffixes=("", "_md")
)

# 5. merge id
"""merged_step5 = pd.merge(
    final_unmatched_df,
    movie_data_items,
    on="id",
    how="left",
    suffixes=("", "_md")
)"""

#comine
final_merged = pd.concat([
    merged_step1[~mask_unmatched],  # title+release_year
    merged_step2[~mask_still_unmatched],  # title+mod_year
    merged_step3[~mask_final_unmatched],  # title
    merged_step4[merged_step4["imdb_id"].notna()]  # imdbId
    #merged_step5  # id or still unmatched
], axis=0)

# remove duplicates
columns_to_drop = [col for col in final_merged.columns if col.endswith('_md')]
final_merged = final_merged.drop(columns=columns_to_drop)
failed_merge = final_merged[final_merged['imdbId'].isna()]
final_merged.columns

# %%
final_merged[['title', 'imdb_id', 'imdbId','data_id']]

# %%
failed_merge

# %%
imdb_ratings = pd.read_csv("imdb/IMDb ratings.csv", sep=',', encoding="latin1")
imdb_ratings['imdb_id'] = imdb_ratings['imdb_title_id'].str.replace('^tt', '', regex=True).astype('Int64')
imdb_ratings = imdb_ratings.drop('imdb_title_id', axis=1)


# %%
imdb_ratings.head()

# %%
merged_with_ratings = pd.merge(
    final_merged,
    imdb_ratings,
    left_on='imdbId',
    right_on='imdb_id',
    how='left',
    suffixes=('', '_imdb')
)

# Find unmatched rows (movies without IMDB ratings)
unmatched_ratings = merged_with_ratings[merged_with_ratings['imdb_id'].isna()]

# %%
merged_with_ratings.columns

# %% [markdown]
# # Baseline Model
# Pick movies based on user demographic

# %%
ratings = pd.read_csv("ml-100k/u1.base", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
ratings_test = pd.read_csv("ml-100k/u1.test", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
#ratings = ratings.drop(columns=["timestamp"])
users = pd.read_csv("ml-100k/u.user", sep="|", names=["userId", "age", "gender", "occupation", "zipcode"])

"""ratings_with_demographics = pd.merge(ratings, users[["userId", "age", "gender", 'zipcode', "occupation"]], on="userId", how='left')
test_ratings_with_demographics = pd.merge(ratings_test, users[["userId", "age", "gender", 'zipcode', "occupation"]], on="userId", how='left')

#Map zipcodes to state
nomi = pgeocode.Nominatim('us')  # US zipcode database

def get_state_from_zip(zipcode):
    try:
        zip_clean = ''.join(filter(str.isdigit, str(zipcode)))[:5]
        if len(zip_clean) == 5:
            location_data = nomi.query_postal_code(zip_clean)
            return location_data.state_code if not pd.isna(location_data.state_code) else 'Unknown'
    except:
        pass
    return 'Unknown'

ratings_with_demographics['state'] = ratings_with_demographics['zipcode'].apply(get_state_from_zip)
test_ratings_with_demographics['state'] = test_ratings_with_demographics['zipcode'].apply(get_state_from_zip)

age_bins = [0, 18, 30, 45, 100]
age_labels = ["0", "18", "30", "45"]

ratings_with_demographics["age_group"] = pd.cut(
    ratings_with_demographics["age"],
    bins=age_bins,
    labels=age_labels,
    right=False
)
test_ratings_with_demographics["age_group"] = pd.cut(
    test_ratings_with_demographics["age"],
    bins=age_bins,
    labels=age_labels,
    right=False
)
"""

# %%
#ratings_with_demographics.head()
#ratings_with_demographics.to_csv('ratings_with_demographics.csv', index=False)
#test_ratings_with_demographics.to_csv('test_ratings_with_demographics.csv', index=False)
ratings_with_demographics = pd.read_csv('ratings_with_demographics.csv')
test_ratings_with_demographics = pd.read_csv('test_ratings_with_demographics.csv')

# %%
ratings_with_demographics[['rating', 'age', 'age_group']].corr()

# %%
#Scale ratings properly
keep_columns = ['vote_count', 'top1000_voters_votes',
       'us_voters_votes', 'non_us_voters_votes', '0_votes', '18_votes', '30_votes', '45_votes', 'F45_votes', 'F30_votes', 'F18_votes', 'F0_votes', 'F_votes','M0_votes', 'M_votes', 'M18_votes', 'M30_votes','M45_votes']

merged_with_ratings = merged_with_ratings.rename(columns={
       'allgenders_0age_avg_vote': '0_avg',
       'allgenders_0age_votes': '0_votes',
       'allgenders_18age_avg_vote': '18_avg',
       'allgenders_18age_votes': '18_votes',
       'allgenders_30age_avg_vote': '30_avg',
       'allgenders_30age_votes': '30_votes',
       'allgenders_45age_avg_vote': '45_avg',
       'allgenders_45age_votes': '45_votes',
       'males_allages_avg_vote': 'M_avg',
       'males_allages_votes': 'M_votes',
       'males_0age_avg_vote': 'M0_avg',
       'males_0age_votes' : 'M0_votes',
       'males_18age_avg_vote': 'M18_avg', 
       'males_18age_votes': 'M18_votes', 
       'males_30age_avg_vote': 'M30_avg',
       'males_30age_votes': 'M30_votes', 
       'males_45age_avg_vote': 'M45_avg', 
       'males_45age_votes': 'M45_votes',
       'females_0age_avg_vote': 'F0_avg',
       'females_0age_votes': 'F0_votes',
       'females_18age_avg_vote': 'F18_avg',
       'females_18age_votes': 'F18_votes',
       'females_30age_avg_vote': 'F30_avg',
       'females_30age_votes': 'F30_votes',
       'females_45age_avg_vote': 'F45_avg',
       'females_45age_votes': 'F45_votes',
       'females_allages_avg_vote': 'F_avg',
       'females_allages_votes': 'F_votes',
       'vote_count': '_votes',
       'vote_average': '_avg',
       'weighted_average_vote':'weighted_avg'})

#Ratings are 1-10 so we need to adjust scaling and counting of other columns to be 1-5
def combine_votes(row):
    new_votes = {
        'votes_5': row['votes_10'] + row['votes_9'],
        'votes_4': row['votes_8'] + row['votes_7'],
        'votes_3': row['votes_6'] + row['votes_5'],
        'votes_2': row['votes_4'] + row['votes_3'],
        'votes_1': row['votes_2'] + row['votes_1']
    }
    return pd.Series(new_votes)

votes_columns = ['votes_1', 'votes_2', 'votes_3', 'votes_4', 'votes_5']
merged_with_ratings[votes_columns] = merged_with_ratings.apply(combine_votes, axis=1)

merged_with_ratings = merged_with_ratings.drop(columns=['votes_6', 'votes_7', 'votes_8', 'votes_9', 'votes_10'])

#Scale columns
columns_to_scale = ['vote_average', 'weighted_average_vote',
'mean_vote', 'median_vote', '0_avg', '18_avg',
'30_avg', '45_avg', 'M_avg',
'M0_avg', 'M18_avg', 'M30_avg',
'M45_avg', 'F_avg', 'F0_avg',
'F18_avg', 'F30_avg', 'F45_avg', 'top1000_voters_rating', 'us_voters_rating', 'non_us_voters_rating']

#Minimum is 1, not 0 so scale rather than divide
def scale_rating(rating):
    return (rating - 1) * (4/9) + 1

for col in columns_to_scale:
    if col in merged_with_ratings.columns:
        merged_with_ratings[col] = merged_with_ratings[col].apply(scale_rating)

# %%
def get_demographic(gender, age_group):
    return f"{gender}{age_group}_avg"

def predict_demographic_rating(row, movies_data):
    movie = movies_data[movies_data['id'] == row['movieId']]
    if len(movie) == 0:
        return np.nan
    
    #Rate by demographic
    demo_col = get_demographic(row['gender'], row['age_group'])
    if not pd.isna(movie[demo_col].iloc[0]):
        return movie[demo_col].iloc[0]
    
    #If no rating for gender, age_group use rating of age_group
    age_col = f"{row['age_group']}_avg"
    if age_col in movie.columns and not pd.isna(movie[age_col].iloc[0]):
        return movie[age_col].iloc[0]
    
    #just gender
    gender_col = f"{row['gender']}_avg"
    if gender_col in movie.columns and not pd.isna(movie[gender_col].iloc[0]):
        return movie[gender_col].iloc[0]
    
    return np.nan

def baseline_model(test_df, movie_demographics):
    predictions = []
    actuals = []
    missing = 0
    
    for _, row in test_df.iterrows():
        pred = predict_demographic_rating(row, movie_demographics)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(row['rating'])
        else:
            missing += 1
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    coverage = 1 - (missing / len(test_df))
    print(f"Coverage: {coverage:.2%}")
    return rmse, predictions, actuals

# %%
train_ratings, temp_ratings = train_test_split(ratings_with_demographics, test_size=0.3, random_state=42)

val_ratings, test_ratings = train_test_split(temp_ratings, test_size=0.33, random_state=42)
"""
def df_to_dataset(df):
    features = {
        "userId": tf.cast(df.userId.values, tf.int32),
        "movieId": tf.cast(df.movieId.values, tf.int32),
        "age": tf.cast(df.age.values, tf.int32),
        "age_group": tf.cast(df.movieId.values, tf.int32),
        "gender": df.gender,
        "occupation": df.occupation,
        "state": df.state
    }
    labels = tf.cast(df.rating.values, tf.float32)
    return tf.data.Dataset.from_tensor_slices((features, labels))

# Create TF datasets
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = df_to_dataset(train_ratings).shuffle(len(train_ratings)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_dataset = df_to_dataset(val_ratings).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = df_to_dataset(test_ratings).batch(BATCH_SIZE).prefetch(AUTOTUNE)"""



# %%
test_ratings

# %%
val_rmse, val_preds, val_actuals = baseline_model(val_ratings, merged_with_ratings)
test_rmse, test_preds, test_actuals = baseline_model(test_ratings, merged_with_ratings)

print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(test_actuals, test_preds, alpha=0.5)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Demographic Baseline: Predicted vs Actual Ratings')
plt.show()

# %%
#General prediction eval function

def evaluate_predictions(actuals, predictions):
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    bias = np.mean(np.array(predictions) - np.array(actuals))
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Bias: {bias:.4f}")

# %%
evaluate_predictions(test_actuals, test_preds)

# %% [markdown]
# So this is still pretty widespread, to get a better baseline incorporate information we might have about a given user, say his/her past rating genre biases

# %%
genres = ["Action", 
    "Adventure", 
    "Animation", 
    "Children's", 
    "Comedy", 
    "Crime", 
    "Documentary", 
    "Drama", 
    "Fantasy", 
    "Film-Noir", 
    "Horror", 
    "Musical", 
    "Mystery", 
    "Romance", 
    "Sci-Fi", 
    "Thriller", 
    "War", 
    "Western"]

def find_user_genre_biases(train_data, movies_data):
    user_genre_biases = {}
    global_mean = train_data['rating'].mean()
    
    for genre in genres:
        #Find movies in genre
        genre_mask = movies_data[genre] == 1
        genre_movies = movies_data[genre_mask]['id']
        
        #If possible generate user genre-bias
        genre_ratings = train_data[train_data['movieId'].isin(genre_movies)]
        user_biases = {}
        
        for user_id in genre_ratings['userId'].unique():
            user_genre_ratings = genre_ratings[genre_ratings['userId'] == user_id]
            if len(user_genre_ratings) > 0:  #if user did rate in this genre
                user_mean = user_genre_ratings['rating'].mean()
                user_biases[user_id] = user_mean / global_mean
            
        user_genre_biases[genre] = user_biases
    
    return user_genre_biases

def predict_demographic_rating_with_genre_bias(row, movies_data, user_genre_biases):
    movie = movies_data[movies_data['id'] == row['movieId']]
    if len(movie) == 0:
        return np.nan
    
    # previous mthd of gender, age_group
    demo_col = get_demographic(row['gender'], row['age_group'])
    base_pred = None
    
    if not pd.isna(movie[demo_col].iloc[0]):
        base_pred = movie[demo_col].iloc[0]
    else:
        #Fallback to just age group
        age_col = f"{row['age_group']}_avg"
        if age_col in movie.columns and not pd.isna(movie[age_col].iloc[0]):
            base_pred = movie[age_col].iloc[0]
        else:
            #Just gender
            gender_col = f"{row['gender']}_avg"
            if gender_col in movie.columns and not pd.isna(movie[gender_col].iloc[0]):
                base_pred = movie[gender_col].iloc[0]
    
    if base_pred is None:
        return np.nan
        
    #Apply users' genre bias
    genre_scaling = 1.0
    genre_count = 0
    
    for genre in genres:
        if movie[genre].iloc[0] == 1:  #If movie in genre
            if (genre in user_genre_biases and 
                row['userId'] in user_genre_biases[genre]):
                genre_scaling *= user_genre_biases[genre][row['userId']]
                genre_count += 1
    
    if genre_count > 0:
        #Take mean over genres
        genre_scaling = genre_scaling ** (1/genre_count)
        return base_pred * genre_scaling
    
    return base_pred

def bias_baseline_model(train_df, test_df, movie_demographics):
    user_genre_biases = find_user_genre_biases(train_df, movie_demographics)
    
    predictions = []
    actuals = []
    missing = 0
    
    for _, row in test_df.iterrows():
        pred = predict_demographic_rating_with_genre_bias(row, movie_demographics, user_genre_biases)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(row['rating'])
        else:
            missing += 1
    
    coverage = 1 - (missing / len(test_df))
    print(f"Coverage: {coverage:.2%}")
    evaluate_predictions(actuals, predictions)
    return predictions, actuals

print("\nGenre-Bias Baseline Results:")
# Use info from train/val ratings
bias_val_preds, bias_val_actuals = bias_baseline_model(
    train_ratings, 
    val_ratings, 
    merged_with_ratings
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.scatter(val_actuals, val_preds, alpha=0.5)
ax1.set_xlabel('Actual Ratings')
ax1.set_ylabel('Predicted Ratings')
ax1.set_title('Baseline')

ax2.scatter(bias_val_actuals, bias_val_preds, alpha=0.5)
ax2.set_xlabel('Actual Ratings')
ax2.set_ylabel('Predicted Ratings')
ax2.set_title('Genre-Bias Informed Baseline')

plt.tight_layout()
plt.show()

# %% [markdown]
# So we definitely have improved slightly. Notice in the graph of actual vs predicted ranges, the distributions while still widespread approximate a linear regression fitting line. We also improved on our metrics.

# %%
train_ratings

# %% [markdown]
# ## Data Exploration

# %%
merged_with_ratings.columns

# %%
#View ratings by genre
genre_ratings = {}
for genre in genres:
    genre_mask = movies_lens_items[genre] == 1
    genre_data = ratings_with_demographics[ratings_with_demographics['movieId'].isin(movies_lens_items[genre_mask]['id'])]
    genre_ratings[genre] = genre_data['rating'].mean()

plt.figure(figsize=(12, 6))
plt.bar(genre_ratings.keys(), genre_ratings.values())
plt.xticks(rotation=45, ha='right')
plt.title('Average rating by genre')
plt.ylabel('Average rating')
plt.tight_layout()
plt.show()

# %% [markdown]
# No definitively better genres, though fantasy and horror tend to rate lower, and noir and war higher.

# %%
# By genre by demograpihc
def plot_genre_demographic_ratings(data, genres):
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(genres))
    width = 0.15
    
    demographics = [('M', '18'), ('M', '30'), ('M', '45'), ('F', '18'), ('F', '30'), ('F', '45')]
    for i, (gender, age) in enumerate(demographics):
        means = []
        for genre in genres:
            genre_mask = data[genre] == 1
            demo_mask = (ratings_with_demographics['gender'] == gender) & \
                       (ratings_with_demographics['age_group'] == age)
            genre_demo_ratings = ratings_with_demographics[demo_mask & \
                ratings_with_demographics['movieId'].isin(data[genre_mask]['id'])]
            means.append(genre_demo_ratings['rating'].mean())
        
        ax.bar(x + i*width, means, width, label=f'{gender}-{age}')
    
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(genres, rotation=45, ha='right')
    ax.set_ylabel('Average Rating')
    ax.set_title('Genre Ratings by Demographic')
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_genre_demographic_ratings(movies_lens_items, genres)

# %%
#By user occupation
occupation_ratings = ratings_with_demographics.merge(users[['userId', 'occupation']], on='userId')
occupation_means = occupation_ratings.groupby('occupation')['rating'].agg(['mean', 'count']).round(2)

plt.figure(figsize=(12, 6))
plt.bar(occupation_means.index, occupation_means['mean'])
plt.xticks(rotation=45, ha='right')
plt.title('Average rating by occupation')
plt.ylabel('Average rating')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
occupation_stats = occupation_ratings.groupby('occupation')['rating'].agg(['mean', 'std']).round(3)
plt.errorbar(occupation_stats.index, occupation_stats['mean'], 
            yerr=occupation_stats['std'], fmt='o')
plt.xticks(rotation=45, ha='right')
plt.title('Average rating by occupation w/ std')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

# %%
#Use state info to visulize
plt.figure(figsize=(15, 6))
state_counts = ratings_with_demographics['state'].value_counts()
plt.bar(state_counts.index, state_counts.values)
plt.xticks(rotation=90)
plt.title('Distribution of ratings by state')
plt.xlabel('State')
plt.ylabel('# ratings')
plt.tight_layout()
plt.show()

# Show avg rating by state
state_ratings = ratings_with_demographics.groupby('state')['rating'].agg(['mean', 'count', 'std']).round(3)
state_ratings = state_ratings.sort_values('mean', ascending=False)

print("\nAverage ratings by state:")
print(state_ratings)

plt.figure(figsize=(15, 6))
plt.errorbar(state_ratings.index, state_ratings['mean'], 
            yerr=state_ratings['std'], fmt='o')
plt.xticks(rotation=90)
plt.title('Average rating by state w/ std')
plt.xlabel('State')
plt.ylabel('Avg rating')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# So for embedding users, definitely is importance in using state information! zipcode will be too overfit

# %%
#Visualize # of reviews per user and color based on rating - does more ratings~with higher/lower score?
user_stats = ratings_with_demographics.groupby('userId').agg({
    'rating': ['count', 'mean']
}).reset_index()
user_stats.columns = ['userId', 'num_reviews', 'avg_rating']

#sort by # review
user_stats = user_stats.sort_values('num_reviews', ascending=True)

min_rating = user_stats['avg_rating'].min()
max_rating = user_stats['avg_rating'].max()
norm = plt.Normalize(min_rating, max_rating)
cmap = plt.cm.RdYlGn

fig, ax = plt.subplots(figsize=(16, 9))
bars = ax.bar(range(len(user_stats)), user_stats['num_reviews'], 
             color=cmap(norm(user_stats['avg_rating'])))

#view average number reviews
mean_reviews = user_stats['num_reviews'].mean()
plt.axhline(y=mean_reviews, color='red', linestyle='--', label=f'Mean Reviews: {mean_reviews:.1f}')

plt.title('Number of Reviews per User (Colored by Average Rating)')
plt.xlabel('Users')
plt.ylabel('Number of Reviews')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Average Rating')

plt.legend()
plt.tight_layout()
plt.show()

print(f"Avg # of reviews/user: {mean_reviews:.2f}")
print(f"Correlation between # of reviews and avg rating: {user_stats['num_reviews'].corr(user_stats['avg_rating']):.3f}")

# %% [markdown]
# There is a slight negative correlation between number of review and average given rating. Also clear outliers which we can ignore when we embed users by specifying math path length.

# %%
keywords_df = pd.read_csv("movie-data/keywords.csv")

merged_with_keywords = pd.merge(
    merged_with_ratings,
    keywords_df,
    left_on="data_id",
    right_on="id",
    how="left"
)

# %%
#extract keyword 'tuples' from json
def parse_str(string):
    try:
        return ast.literal_eval(string)
    except:
        return []

merged_with_keywords['keywords'] = merged_with_keywords["keywords"].apply(parse_str)
merged_with_keywords['production_companies'] = merged_with_keywords['production_companies'].apply(parse_str)
merged_with_keywords['production_countries'] = merged_with_keywords["production_countries"].apply(parse_str)

#extract list of keywords
merged_with_keywords["keywords"] = merged_with_keywords['keywords'].apply(
    lambda x: [kw["name"] for kw in x] if x else []
)
merged_with_keywords['production_companies'] = merged_with_keywords['production_companies'].apply(
    lambda x: [kw["name"] for kw in x] if x else []
)
merged_with_keywords['production_countries'] = merged_with_keywords['production_countries'].apply(
    lambda x: [kw["name"] for kw in x] if x else []
)

# %%
merged_with_keywords = merged_with_keywords.drop(columns=['genres', 'original_language', 'spoken_languages'])
merged_with_keywords.columns

# %%
merged_with_keywords[['budget', 'data_id', 'imdb_id',
       'overview', 'popularity', 'production_companies',
       'production_countries', 'revenue', 'runtime',
       'tagline']].head()

# %%


# %% [markdown]
# # Collaborative Filtering Model
# Since we have so much non-linear information about connection between movies, users, and between both groups we will create latent embeddings for both and then use a NN to match across for a collaboritve filter type system.

# %%
#First embed movie attributes

movie_unembedded = merged_with_keywords.copy()

#handle numerix columns
"""numerics = ['revenue', 'runtime', 'budget', 'popularity'
       'vote_average', 'vote_count', 'weighted_average_vote',
       'total_votes', 'mean_vote', 'median_vote', 'votes_5', 'votes_4',
       'votes_3', 'votes_2', 'votes_1', '0_avg', '0_votes', '18_avg',
       '18_votes', '30_avg', '30_votes', '45_avg', '45_votes', 'M_avg',
       'M_votes', 'M0_avg', 'M0_votes', 'M18_avg', 'M18_votes', 'M30_avg',
       'M30_votes', 'M45_avg', 'M45_votes', 'F_avg', 'F_votes', 'F0_avg',
       'F0_votes', 'F18_avg', 'F18_votes', 'F30_avg', 'F30_votes', 'F45_avg',
       'F45_votes', 'top1000_voters_rating', 'top1000_voters_votes',
       'us_voters_rating', 'us_voters_votes', 'non_us_voters_rating',
       'non_us_voters_votes']"""

#normalize/scale pure numerics:
movie_unembedded['revenue_norm'] = np.log1p(movie_unembedded['revenue'])
movie_unembedded['budget_norm'] = np.log1p(movie_unembedded['budget'])
movie_unembedded['popularity_norm'] = np.log1p(movie_unembedded['popularity'])

scaler = StandardScaler()
movie_unembedded[['revenue_norm', 'runtime', 'budget_norm', 'popularity_norm']] = scaler.fit_transform(
    movie_unembedded[['revenue_norm', 'runtime', 'budget_norm', 'popularity_norm']]
)

#vote rating columns can be kept as is

#to not overbias movies with more reviews, normalize votes for each category by # votes given in category, that is how confideent are we in given ratings?
mean = movie_unembedded["weighted_avg"].mean()
m=100
avg_cols = [col for col in movie_unembedded.columns if col.endswith('_avg')]

for avg_col in avg_cols:
    #corresponding total vote column
    votes_col = avg_col.replace('_avg', '_votes')
    
    if not votes_col in movie_unembedded.columns:
        continue

    normalized_col = avg_col.replace('_avg', '_norm')
    votes = movie_unembedded[votes_col]
    avgs = movie_unembedded[avg_col]
    movie_unembedded[normalized_col] = (votes / (votes + m)) * avgs + (mean / (votes + m)) * mean

#For year just normalize on bounds
movie_unembedded['release_year_norm'] = (movie_unembedded['release_year'] - movie_unembedded['release_year'].min()) / (movie_unembedded['release_year'].max() - movie_unembedded['release_year'].min())
movie_unembedded['keywords_text'] = movie_unembedded['keywords'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '').tolist()


# %%
movie_unembedded['keywords_text']

# %%
#embed text per movie

word2vec = api.load("word2vec-google-news-300")#use word2vec for preembedded words

texts = movie_unembedded['overview'].fillna('').tolist() + movie_unembedded['tagline'].fillna('').tolist() + movie_unembedded['keywords_text'].fillna('').tolist()

#tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

#sequence for tf
overview_seqs = tokenizer.texts_to_sequences(movie_unembedded['overview'].fillna(''))
tagline_seqs = tokenizer.texts_to_sequences(movie_unembedded['tagline'].fillna(''))
keywords_seqs = tokenizer.texts_to_sequences(movie_unembedded['keywords_text'].fillna(''))

#pad for lstm
max_len = 200
overview_padded = pad_sequences(overview_seqs, maxlen=max_len)
tagline_padded = pad_sequences(tagline_seqs, maxlen=max_len)
keywords_padded = pad_sequences(keywords_seqs, maxlen=max_len)

#embed
embedding_dim = 300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

#find embedding in word2vec
for word, i in word_index.items():
    if word in word2vec:
        embedding_matrix[i] = word2vec[word]


embedding_layer = Embedding(
    input_dim=len(word_index) + 1,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_len,
    trainable=False
)
#create seperate LSTM embedding for each feature
overview_processor = Sequential([
    Input(shape=(max_len,)),
    embedding_layer,
    LSTM(128)
])

#tagline and keywords are shorter so we adjust LSTM accordingly
tagline_processor = Sequential([
    Input(shape=(max_len,)),
    embedding_layer,
    LSTM(64)
])

keywords_processor = Sequential([
    Input(shape=(max_len,)),
    embedding_layer,
    LSTM(64)
])

#Now take these embeddings to create overall textual features embedding
final_processor = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu')
])

overview_input = Input(shape=(max_len,))
tagline_input = Input(shape=(max_len,))
keywords_input = Input(shape=(max_len,))

overview_features = overview_processor(overview_input)
tagline_features = tagline_processor(tagline_input)
keywords_features = keywords_processor(keywords_input)

combined = Concatenate()([overview_features, tagline_features, keywords_features])
movie_embedding = final_processor(combined)

movie_text_embedder = Model(
    inputs=[overview_input, tagline_input, keywords_input],
    outputs=movie_embedding
)

embedding_output = movie_text_embedder([
    overview_padded,
    tagline_padded,
    keywords_padded
], training=False)

# %%
embedding_output

# %%
#now create overall movie embedding

embedding_dim=64

numeric_features = ['runtime', 'revenue_norm', 'budget_norm', 'popularity_norm', 'release_year_norm',
    '0_norm', '18_norm', '30_norm', '45_norm',
    'M_norm', 'M0_norm', 'M18_norm', 'M30_norm',
    'M45_norm', 'F_norm', 'F0_norm', 'F18_norm',
    'F30_norm', 'F45_norm', 'weighted_avg', 'total_votes'
]
#genres
movies_numeric = movie_unembedded[numeric_features].fillna(0).values.astype(np.float32)
movies_genres = movie_unembedded[genres].values.astype(np.float32)

text_input = Input(shape=(embedding_output.shape[1],))
numeric_input = Input(shape=(movies_numeric.shape[1],))
genre_input = Input(shape=(movies_genres.shape[1],))

numeric_dense = Dense(64, activation='relu')(numeric_input)
genre_dense = Dense(32, activation='relu')(genre_input)

combined_features = Concatenate()([text_input, numeric_dense, genre_dense])
final_dense = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(embedding_dim, activation='relu')
])

final_embedding = final_dense(combined_features)

movie_full_embedder = Model(
    inputs=[text_input, numeric_input, genre_input],
    outputs=final_embedding
)

embeddings = movie_full_embedder([
    embedding_output,
    movies_numeric,
    movies_genres
], training=False)

movie_items_embedded = {}
for movie_id, embedding in zip(movie_unembedded['id_x'], embeddings):
    movie_items_embedded[movie_id] = embedding.numpy()

# %% [markdown]
# To ensure we have a good embedding of movies, run PCA

# %%
#makes the most sense to color by age_group
def visualize_embeddings(embeds, groups, kind):
    pca = PCA(n_components=2) #2d
    pca = pca.fit_transform(embeds)


    reducer = umap.UMAP(n_components=2, random_state=42)
    umaper = reducer.fit_transform(embeds)

    unique_groups = np.unique(groups)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_groups)))


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))

    for i, group in enumerate(unique_groups):
        idx = groups == group
        ax1.scatter(pca[idx, 0], pca[idx, 1], 
                    color=colors[i], label=f'{unique_groups[group]}', alpha=0.6, s=15)
        ax2.scatter(umaper[idx, 0], umaper[idx, 1],
                    color=colors[i], label=f'{unique_groups[group]}', alpha=0.6, s=15)

    ax1.legend()
    ax1.set_title(f"PCA {kind} Embeddings")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")

    ax2.legend()
    ax2.set_title(f"UMAP {kind} Embeddings")
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")

    plt.tight_layout()
    plt.show()

genre_indices = np.argmax(movie_unembedded[genres].values, axis=1)
visualize_embeddings(embeddings, genre_indices, 'Movie')
visualize_embeddings(embedding_output, genre_indices, 'Movie Text')
"""

pca1 = PCA(n_components=2) #2d
movie_items_pca = pca1.fit_transform(embeddings)

pca2 = PCA(n_components=2)
movie_text_pca = pca2.fit_transform(embedding_output)

reducer1 = umap.UMAP(n_components=2, random_state=42)
movie_items_umap = reducer1.fit_transform(embeddings)

reducer2 = umap.UMAP(n_components=2, random_state=42)
movie_text_umap = reducer2.fit_transform(embedding_output)

genre_indices = np.argmax(movie_unembedded[genres].values, axis=1)
unique_genres = np.unique(genre_indices)
colors = plt.cm.get_cmap('tab20', len(unique_genres))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))

for i, genre_id in enumerate(unique_genres):
    idx = genre_indices == genre_id
    ax1.scatter(movie_items_pca[idx, 0], movie_items_pca[idx, 1], 
                color=colors(i), label=genres[genre_id], alpha=0.6, s=15)
    ax2.scatter(movie_text_pca[idx, 0], movie_text_pca[idx, 1], 
                color=colors(i), label=genres[genre_id], alpha=0.6, s=15)
    ax3.scatter(movie_items_umap[idx, 0], movie_items_umap[idx, 1],
                color=colors(i), label=genres[genre_id], alpha=0.6, s=15)
    ax4.scatter(movie_text_umap[idx, 0], movie_text_umap[idx, 1],
                color=colors(i), label=genres[genre_id], alpha=0.6, s=15)

ax1.legend()
ax1.set_title("PCA Movie Embeddings")
ax1.set_xlabel("Component 1")
ax1.set_ylabel("Component 2")

ax2.legend()
ax2.set_title("PCA Movie Text Embeddings")
ax2.set_xlabel("Component 1")
ax2.set_ylabel("Component 2")

ax3.legend()
ax3.set_title("UMAP Movie Embeddings")
ax3.set_xlabel("UMAP Dim 1")
ax3.set_ylabel("UMAP Dim 2")

ax4.legend()
ax4.set_title("UMAP Movie Text Embeddings")
ax4.set_xlabel("UMAP Dim 1")
ax4.set_ylabel("UMAP Dim 2")

plt.tight_layout()
plt.show()

train_ratings['age_group'].values"""



# %% [markdown]
# Not sure what do make of this but text has a similar, yet random embedding in both. There does seem to be more of some rhyme/reason to movie embeddings.

# %% [markdown]
# ## ALS

# %%
user_id_map = {uid: idx+1 for idx, uid in enumerate(train_ratings['userId'].unique())}
movie_id_map = {mid: idx+1 for idx, mid in enumerate(sorted(list(movie_items_embedded.keys())))}
num_users = len(user_id_map)
num_movies = len(movie_id_map)

# %%
ratings_with_demographics.columns

# %%
from tqdm import tqdm
from scipy.optimize import nnls
import scipy.sparse as sp

def train_nonneg_als(ratings_df, user_id_map, movie_id_map, n_factors=64, 
                     reg_param=0.1, max_iter=15, tol=1e-4, verbose=False):
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    
    user_factors = np.random.uniform(0, 0.1, (num_users + 1, n_factors))# +1 for each for unknown
    movie_factors = np.random.uniform(0, 0.1, (num_movies + 1, n_factors))
    
    #biases at zero
    user_bias = np.zeros(num_users + 1)
    movie_bias = np.zeros(num_movies + 1)
    global_bias = ratings_df['rating'].mean()
    
    #convert ratings to sparse matrix
    rows = [user_id_map.get(uid, 0) for uid in ratings_df['userId']]
    cols = [movie_id_map.get(mid, 0) for mid in ratings_df['movieId']]
    values = ratings_df['rating'].values
    
    M = sp.csr_matrix((values, (rows, cols)), 
                      shape=(num_users + 1, num_movies + 1))
    
    # mask for existing ratings to use while fix
    mask = sp.csr_matrix(
        (np.ones_like(values), (rows, cols)),
        shape=(num_users + 1, num_movies + 1)
    )
    
    # alternate freezing and optimization
    prev_loss = float('inf')
    for iteration in range(max_iter):
        if verbose:
            print(f"Iteration {iteration+1}/{max_iter}")
            
        #fix movie factors, solve for user-rating
        for u in tqdm(range(1, num_users + 1), disable=not verbose):
            movie_rated = mask[u].indices
            if len(movie_rated) == 0:
                continue
                
            ratings = M[u, movie_rated].toarray().flatten()
            Y_u = movie_factors[movie_rated]
            
            #incorporate bias
            ratings_adj = ratings - global_bias - movie_bias[movie_rated]
            
            # Solve non-negative least squares
            A = Y_u.T @ Y_u + reg_param * np.eye(n_factors)
            b = Y_u.T @ ratings_adj
            user_factors[u], _ = nnls(A, b)
            
            # Update bias
            residuals = ratings - (user_factors[u] @ Y_u.T + 
                                 global_bias + movie_bias[movie_rated])
            user_bias[u] = np.mean(residuals)
            
        #fix user-rating, solve for movie
        for i in tqdm(range(1, num_movies + 1), disable=not verbose):
            users_rated = mask[:, i].indices
            if len(users_rated) == 0:
                continue
                
            ratings = M[users_rated, i].toarray().flatten()
            X_i = user_factors[users_rated]
            
            ratings_adj = ratings - global_bias - user_bias[users_rated]
            
            A = X_i.T @ X_i + reg_param * np.eye(n_factors)
            b = X_i.T @ ratings_adj
            movie_factors[i], _ = nnls(A, b)
            
            #adjust movie bias
            residuals = ratings - (X_i @ movie_factors[i] + 
                                 global_bias + user_bias[users_rated])
            movie_bias[i] = np.mean(residuals)
            
        #Check convergence
        train_rmse = calculate_rmse(ratings_df, user_factors, movie_factors,
                                  user_bias, movie_bias, global_bias,
                                  user_id_map, movie_id_map)
        
            
        """if abs(prev_loss - train_rmse) < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break
            
        prev_loss = train_rmse"""
        
    return user_factors, movie_factors, user_bias, movie_bias, global_bias


# %%

def predict_ratings(user_ids, movie_ids, user_factors, movie_factors,
                   user_bias, movie_bias, global_bias, 
                   user_id_map, movie_id_map):
    predictions = []
    
    for user_id, movie_id in zip(user_ids, movie_ids):
        user_idx = user_id_map.get(user_id, 0)
        movie_idx = movie_id_map.get(movie_id, 0)
        
        if user_idx == 0 or movie_idx == 0:
            pred = global_bias
        else:
            pred = (user_factors[user_idx] @ movie_factors[movie_idx] +
                   global_bias + user_bias[user_idx] + movie_bias[movie_idx])
        
        predictions.append(np.clip(pred, 1.0, 5.0))#clip at 1-5
        
    return np.array(predictions)

def calculate_rmse(ratings_df, user_factors, movie_factors,
                  user_bias, movie_bias, global_bias,
                  user_id_map, movie_id_map):
    predictions = predict_ratings(
        ratings_df['userId'], 
        ratings_df['movieId'],
        user_factors, movie_factors,
        user_bias, movie_bias, global_bias,
        user_id_map, movie_id_map
    )
    return np.sqrt(mean_squared_error(ratings_df['rating'], predictions))

# %%
user_factors, movie_factors, user_bias, movie_bias, global_bias = train_nonneg_als(
    train_ratings,
    user_id_map,
    movie_id_map,
    n_factors=64,
    verbose=True
)

als_test_predictions = predict_ratings(
    test_ratings['userId'],
    test_ratings['movieId'],
    user_factors, 
    movie_factors,
    user_bias,
    movie_bias, 
    global_bias,
    user_id_map,
    movie_id_map
)


test_rmse = np.sqrt(mean_squared_error(test_ratings['rating'], als_test_predictions))
print(f"Test RMSE: {test_rmse:.4f}")

u1_als_test_predictions = predict_ratings(
    test_ratings_with_demographics['userId'],
    test_ratings_with_demographics['movieId'],
    user_factors, 
    movie_factors,
    user_bias,
    movie_bias, 
    global_bias,
    user_id_map,
    movie_id_map
)

# Evaluate
u1_test_rmse = np.sqrt(mean_squared_error(test_ratings_with_demographics['rating'], u1_als_test_predictions))
print(f"U1.test RMSE: {u1_test_rmse:.4f}")

# %%
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_and_evaluate(model, train_data, val_data, test_data, final_test, epochs=15, batch_size=32,m_factor=False):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )
    
    # For dictionary input data
    if m_factor:
        x_train = [train_data[k] for k in train_data if k != 'targets']
        y_train = train_data['targets']
        
        x_val = [val_data[k] for k in val_data if k != 'targets']
        y_val = val_data['targets']
        
        # Train model
        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Predict on test data
        x_test = [test_data[k] for k in test_data if k != 'targets']
        u1_test = [final_test[k] for k in test_data if k != 'targets']

        test_predictions = model.predict(x_test)
        u1_predictions = model.predict(u1_test)
        test_targets = test_data['targets']
        u1_targets = final_test['targets']
    elif isinstance(train_data, dict) and 'targets' in train_data:
        x_train = []
        for k in train_data:
            if k != 'targets':
                if k in ['user_id', 'movie_id']:
                    x_train.append(tf.cast(train_data[k], tf.int32))
                else:
                    x_train.append(tf.cast(train_data[k], tf.float32))
        
        y_train = tf.cast(train_data['targets'], tf.float32)
        
        # same for validation data
        x_val = []
        for k in val_data:
            if k != 'targets':
                if k in ['user_id', 'movie_id']:
                    x_val.append(tf.cast(val_data[k], tf.int32))
                else:
                    x_val.append(tf.cast(val_data[k], tf.float32))
        
        y_val = tf.cast(val_data['targets'], tf.float32)
        
        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )
        
        #..and test data
        x_test = []
        for k in test_data:
            if k != 'targets':
                if k in ['user_id', 'movie_id']:
                    x_test.append(tf.cast(test_data[k], tf.int32))
                else:
                    x_test.append(tf.cast(test_data[k], tf.float32))
                    
        # final test data
        u1_test = []
        for k in final_test:
            if k != 'targets':
                if k in ['user_id', 'movie_id']:
                    u1_test.append(tf.cast(final_test[k], tf.int32))
                else:
                    u1_test.append(tf.cast(final_test[k], tf.float32))
        
        test_predictions = model.predict(x_test)
        u1_predictions = model.predict(u1_test)
        test_targets = test_data['targets']
        u1_targets = final_test['targets']
    else:
        # For simple input, ie, for matrix factor
        history = model.fit(
            x=train_data['inputs'],
            y=train_data['targets'],
            validation_data=(val_data['inputs'], val_data['targets']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )
        
        test_predictions = model.predict(test_data['inputs'])
        u1_predictions = model.predict(final_test['inputs'])
        test_targets = test_data['targets']
        u1_targets = final_test['targets']
    
    # Calculate metrics
    t_mae = mean_absolute_error(test_targets, test_predictions)
    t_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    
    print(f"Test MAE: {t_mae:.4f}")
    print(f"Test RMSE: {t_rmse:.4f}")

    u1_mae = mean_absolute_error(u1_targets, u1_predictions)
    u1_rmse = np.sqrt(mean_squared_error(u1_targets, u1_predictions))
    
    print(f"U1.Test MAE: {u1_mae:.4f}")
    print(f"U1.Test RMSE: {u1_rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(test_targets, test_predictions, alpha=0.5)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(f'Test Data: Actual vs Predicted (RMSE={t_rmse:.4f})')

    plt.subplot(1, 3, 3)
    plt.scatter(u1_targets, u1_predictions, alpha=0.5)

    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(f'U1: Actual vs Predicted (RMSE={u1_rmse:.4f})')
    
    plt.tight_layout()
    plt.show()
    
    return history, test_predictions, u1_predictions, t_rmse, u1_rmse

# %%
#simple matrix factorization

def mf(num_users, num_movies, embedding_dim=32):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    
    user_embedding = Embedding(num_users + 1, embedding_dim, input_length=1)(user_input)
    movie_embedding = Embedding(num_movies + 1, embedding_dim, input_length=1)(movie_input)
    
    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)
    
    dot_product = Dot(axes=1)([user_vec, movie_vec])
    
    # simple bias
    user_bias = Embedding(num_users + 1, 1, input_length=1)(user_input)
    movie_bias = Embedding(num_movies + 1, 1, input_length=1)(movie_input)
    global_bias = tf.constant([3.5])  #used as 'mean'
    
    # Combine
    output = Add()([dot_product, Flatten()(user_bias), Flatten()(movie_bias), global_bias])
    output = Lambda(lambda x: tf.clip_by_value(x, 1.0, 5.0))(output)
    
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    return model

def prepare_mf(df, user_id_map, movie_id_map):
    user_ids = np.array([user_id_map.get(uid, 0) for uid in df['userId']])
    movie_ids = np.array([movie_id_map.get(mid, 0) for mid in df['movieId']])
    ratings = df['rating'].values
    
    return {
        'inputs': [user_ids, movie_ids],
        'targets': ratings
    }

mf_model = mf(num_users, num_movies)

# Prepare data for MF model
mf_train_data = prepare_mf(train_ratings, user_id_map, movie_id_map)
mf_val_data = prepare_mf(val_ratings, user_id_map, movie_id_map)
mf_test_data = prepare_mf(test_ratings, user_id_map, movie_id_map)
mf_u1_data = prepare_mf(test_ratings_with_demographics, user_id_map, movie_id_map)

# Train and evaluate baseline
_, _, _, mf_t_rmse, mf_u1_rmse = train_and_evaluate(
    mf_model, mf_train_data, mf_val_data, mf_test_data, mf_u1_data, epochs=10, m_factor=True
)

# %% [markdown]
# Simple matrix based performance is etter than initial baseline as expected

# %% [markdown]
# ### Embed User Ratings

# %%
ratings_with_demographics.columns

# %%
def build_user_histories(train_df, movie_items_embedded, embedding_dim):
    user_histories = {}
    train_df_sorted = train_df.sort_values(['userId', 'timestamp'])

    for user_id, group in train_df_sorted.groupby('userId'):
        sequence = []
        for _, row in group.iterrows():
            movie_id = row['movieId']
            if movie_id in movie_items_embedded:
                emb = movie_items_embedded[movie_id]
                rating = row['rating']
                timestamp = row['timestamp']
                sequence.append(np.concatenate([emb, [rating], [timestamp]]))
        user_histories[user_id] = sequence

        #normalize timestamps per user
        for user_id, sequence in user_histories.items():
            if len(sequence) > 0:
                timestamps = np.array([item[-1] for item in sequence])
                min_ts, max_ts = timestamps.min(), timestamps.max()
                time_range = max_ts - min_ts
                if time_range > 0:
                    for i in range(len(sequence)):
                        # Norm 0-1
                        sequence[i][-1] = (sequence[i][-1] - min_ts) / time_range
                else:
                    for i in range(len(sequence)):
                        sequence[i][-1] = 1.0
    return user_histories

def prepare_embedding(df, movie_items_embedded, embedding_dim,
                       categoricals, user_id_map, movie_id_map, user_histories=None, max_seq_len=220):
    
    df = df.reset_index(drop=True)
    demo_features = []
    
    #Normalize age - for non-training it is needed
    if 'age' in df.columns:
        # Normalize age
        age_mean, age_std = df['age'].mean(), df['age'].std()
        df['age_norm'] = (df['age'] - age_mean) / (age_std + 1e-8)
        demo_features.append(df[['age_norm']].fillna(0))
    
    #One-hot encode categorical columns
    for cat_name, cat_data in categoricals.items():
        if cat_name in df.columns:
            cat_dummies = pd.get_dummies(df[cat_name], prefix=cat_name)
            cat_dummies = cat_dummies.reindex(columns=cat_data.columns, fill_value=0)
            cat_dummies.index = df.index
            demo_features.append(cat_dummies)

    if demo_features:
        demo_matrix = pd.concat(demo_features, axis=1).values
    else:
        demo_matrix = np.zeros((len(df), 1))

    user_ids = []
    movie_ids = []
    movie_embs = []
    user_seqs = []
    targets = []

    for idx, row in df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']

        user_idx = user_id_map.get(user_id, 0)
        movie_idx = movie_id_map.get(movie_id, 0)  # unseen users mapped to 0

        #Build rating histories during train/val only
        if user_histories is not None: #past user ratings passed in during testing
            hist = user_histories.get(user_id, [])
        else:
            hist_df = df[(df['userId'] == user_id) & (df.index < idx)]
            hist = []
            for _, h_row in hist_df.iterrows():
                m = h_row['movieId']
                if m in movie_items_embedded:
                    emb = movie_items_embedded[m]
                    r = h_row['rating']
                    t = h_row['timestamp']
                    hist.append(np.concatenate([emb, [r], [t]]))

        if len(hist) > 0:
            if len(hist) > max_seq_len:
                    sequence = hist[-max_seq_len:]  # Keep most recent
            else:
                pad_len = max_seq_len - len(hist)
                # Zero padding at the beginning so keep later ratings
                sequence = [[0.] * (embedding_dim + 2)] * pad_len + hist

        else:
            # No history
            sequence = [[0.] * (embedding_dim + 2)] * max_seq_len

        movie_emb = movie_items_embedded.get(movie_id, np.zeros(embedding_dim))
        
        #add to lists
        user_ids.append(user_idx)
        movie_ids.append(movie_idx)
        movie_embs.append(movie_emb)
        user_seqs.append(sequence)
        targets.append(row['rating'])

    user_ids = np.array(user_ids, dtype=np.int32)  # For TensorFlow embedding layers
    movie_ids = np.array(movie_ids, dtype=np.int32)
    user_seqs = np.array(user_seqs, dtype=np.float32)  # For LSTM/Attention layers
    demo_matrix = np.array(demo_matrix, dtype=np.float32)  # For dense layers
    movie_embs = np.array(movie_embs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    # Convert to numpy arrays
    return {
        'user_id': np.array(user_ids),
        'movie_id': np.array(movie_ids),
        'user_seq': np.array(user_seqs),
        'user_demo': demo_matrix,
        'movie_emb': np.array(movie_embs),
        'targets': np.array(targets)
    }


# %%
#Normalize age
"""age_scaler = StandardScaler()
age_scaler.fit(train_ratings[['age']])

#One-hot encode columns
gender_categorical_train = pd.get_dummies(train_ratings['gender'], prefix='gender')
state_categorical_train = pd.get_dummies(train_ratings['state'], prefix='state')
age_group_categorical_train = pd.get_dummies(train_ratings['age_group'], prefix='age_group')
#occupation_categorical_train = pd.get_dummies(train_ratings['occupation'], prefix='occupation')

categoricals = {
    'gender': gender_categorical_train,
    'state': state_categorical_train,
    'age_group': age_group_categorical_train#,
    #'occupation': occupation_categorical_train
}"""

#To incorporate userId and timestamp we can need a way to incorporate how a given user has rated overtime
#=> we use a attention to process ratings as sequential

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)
        
    def call(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch_size, seq_len, 1)
        a = K.softmax(e, axis=1)  # (batch_size, seq_len, 1)
        output = x * a  # (batch_size, seq_len, feature_dim)
        return K.sum(output, axis=1)  #(batch, feature_dim)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

#To deal with ratings for users with no or little past ratings (for cold start) we base on their demograhpic


"""train_data = prepare_embedding(
    train_ratings,
    movie_items_embedded,
    embedding_dim,
    age_scaler,
    categoricals,
    user_id_map,
    movie_id_map
    user_histories=None,
    max_seq_len=max_seq_len
)
print('training prepared')"""



# %%
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# easily create model
def create_ncf_model(num_users, num_movies, embedding_dim=64, 
                   demo_dim=0, max_seq_len=220):
    user_id_input = Input(shape=(1,), name='user_id')
    movie_id_input = Input(shape=(1,), name='movie_id')
    
    # embedding bbased inputs
    user_seq_input = Input(shape=(max_seq_len, embedding_dim + 2), name='user_seq')
    user_demo_input = Input(shape=(demo_dim,), name='user_demo')
    movie_emb_input = Input(shape=(embedding_dim,), name='movie_emb')
    
    # incorporate matrix based
    mf_user_embedding = Embedding(num_users + 1, embedding_dim, 
                                name='mf_user_embedding')(user_id_input)
    mf_user_vec = Flatten()(mf_user_embedding)
    
    mf_movie_embedding = Embedding(num_movies + 1, embedding_dim, 
                                 name='mf_movie_embedding')(movie_id_input)
    mf_movie_vec = Flatten()(mf_movie_embedding)
    
    #path embeddings -larger for better rep
    mlp_user_embedding = Embedding(num_users + 1, embedding_dim*2, 
                                 name='mlp_user_embedding')(user_id_input)
    mlp_user_vec = Flatten()(mlp_user_embedding)
    
    mlp_movie_embedding = Embedding(num_movies + 1, embedding_dim*2, 
                                  name='mlp_movie_embedding')(movie_id_input)
    mlp_movie_vec = Flatten()(mlp_movie_embedding)
    
    # feed sequential ratings attention
    attention_layer = Attention()(user_seq_input)
    seq_features = Dense(embedding_dim, activation='relu')(attention_layer)
    
    #add demographic data
    if demo_dim > 0:
        demo_features = Dense(embedding_dim//2, activation='relu')(user_demo_input)
        
        #Combine all features
        user_features = Concatenate()([mlp_user_vec, seq_features, demo_features])
    else:
        user_features = Concatenate()([mlp_user_vec, seq_features])

    
    #process movie features
    movie_content = Dense(embedding_dim, activation='relu')(movie_emb_input)
    movie_features = Concatenate()([mlp_movie_vec, movie_content])
    
    #matrix factorization
    mf_vector = Multiply()([mf_user_vec, mf_movie_vec])
    
    # Final combining user and movies
    mlp_vector = Concatenate()([user_features, movie_features])
    
    # Deep layers with regularization and normalization
    mlp_vector = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(mlp_vector)
    mlp_vector = BatchNormalization()(mlp_vector)
    mlp_vector = Dropout(0.3)(mlp_vector)
    
    mlp_vector = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(mlp_vector)
    mlp_vector = BatchNormalization()(mlp_vector)
    mlp_vector = Dropout(0.2)(mlp_vector)
    
    # Combine MF and MLP paths
    predict_vector = Concatenate()([mf_vector, mlp_vector])
    
    #final output
    prediction = Dense(1, activation='linear')(predict_vector)#use linear
    
    #Clip to valid range
    prediction = Lambda(lambda x: tf.clip_by_value(x, 1.0, 5.0))(prediction)
    
    model = Model(
        inputs=[user_id_input, movie_id_input, user_seq_input, user_demo_input, movie_emb_input],
        outputs=prediction
    )
    
    # Compile with Adam optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

# %%
user_histories = build_user_histories(
    train_ratings, movie_items_embedded, embedding_dim
)
 

# %%
# One-hot encode categorical columns
categoricals = {}
for col in ['gender', 'state', 'age_group']:
    if col in train_ratings.columns:
        cat_dummies = pd.get_dummies(train_ratings[col], prefix=col)
        categoricals[col] = cat_dummies

# use to calcualte demographic dime
demo_dim = sum(df.shape[1] for df in categoricals.values())
if 'age' in train_ratings.columns:
    demo_dim += 1  #Add normalized age feature
 

# %%
   
# Prepare data
train_data = prepare_embedding(
    train_ratings, movie_items_embedded, embedding_dim,
    categoricals, user_id_map, movie_id_map,
    user_histories=None, max_seq_len=220
)
#train_data['user_demo'] = train_data['user_demo'].astype('float64')

val_data = prepare_embedding(
    val_ratings, movie_items_embedded, embedding_dim,
    categoricals, user_id_map, movie_id_map,
    user_histories=user_histories, max_seq_len=220
)
#val_data['user_demo'] = val_data['user_demo'].astype('float64')

test_data = prepare_embedding(
    test_ratings, movie_items_embedded, embedding_dim,
    categoricals, user_id_map, movie_id_map,
    user_histories=user_histories, max_seq_len=220
)
#test_data['user_demo'] = test_data['user_demo'].astype('float64')

u1_test_data = prepare_embedding(
    test_ratings_with_demographics, movie_items_embedded, embedding_dim,
    categoricals, user_id_map, movie_id_map,
    user_histories=user_histories, max_seq_len=220
)
#u1_test_data['user_demo'] = u1_test_data['user_demo'].astype('float64')

# %%
train_ratings.columns

# %%
"""#visualize user embeddings
user_embed_model = Model(
    inputs=[user_id_input, user_seq_input, user_demo_input],
    outputs=user_features
)

# Get user embeddings for visualization
user_embeddings = user_embed_model.predict([
    train_data['user_id'],
    train_data['user_seq'],
    train_data['user_demo']
])

# Convert age groups to indices for visualization
age_groups = train_ratings['age_group'].values
unique_groups = np.unique(age_groups)
age_indices = {group: idx for idx, group in enumerate(unique_groups)}
age_group_indices = np.array([age_indices[g] for g in age_groups])

# Visualize using existing function
visualize_embeddings(user_embeddings, age_group_indices, 'User')"""

# %%
for k, v in train_data.items():
    if k != 'targets':
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")

# %%
test_data.keys()

# %%
ncf_model = create_ncf_model(
    num_users, num_movies, embedding_dim, 
    demo_dim, max_seq_len=220
)

#train +eval
_, test_predictions, u1_test_predictions, ncf_rmse, u1_tmse = train_and_evaluate(
    ncf_model, train_data, val_data, test_data, u1_test_data, epochs=25
)


# %%

#Analyze errors
test_errors = np.abs(test_data['targets'] - test_predictions.flatten())
u1_test_errors = np.abs(u1_test_data['targets'] - u1_test_predictions.flatten())

error_df = pd.DataFrame({
    'user_id': test_data['user_id'].flatten(),
    'movie_id': test_data['movie_id'].flatten(),
    'actual': test_data['targets'],
    'predicted': test_predictions.flatten(),
    'test_error': test_errors
})

u1_error_df = pd.DataFrame({
    'user_id': u1_test_data['user_id'].flatten(),
    'movie_id': u1_test_data['movie_id'].flatten(),
    'actual': u1_test_data['targets'],
    'predicted': u1_test_predictions.flatten(),
    'test_error': u1_test_errors
})

print("\nTest Error distribution:")
print(f"Mean error: {test_errors.mean():.4f}")
print(f"Median error: {np.median(test_errors):.4f}")
print(f"90th percentile error: {np.percentile(test_errors, 90):.4f}")


plt.figure(figsize=(10, 6))
plt.hist(test_errors, bins=20)
plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.title('Distribution of Prediction Test Errors')
plt.tight_layout()
plt.show()

print("\nU1.Test Error distribution:")
print(f"Mean error: {u1_test_errors.mean():.4f}")
print(f"Median error: {np.median(u1_test_errors):.4f}")
print(f"90th percentile error: {np.percentile(u1_test_errors, 90):.4f}")


plt.figure(figsize=(10, 6))
plt.hist(u1_test_errors, bins=20)
plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.title('Distribution of Prediction U1.Test Errors')
plt.tight_layout()
plt.show()

# %% [markdown]
# 

# %% [markdown]
# 

# %%
user_id_input = Input(shape=(1,), name='user_id')

# embedding bbased inputs
user_seq_input = Input(shape=(220, embedding_dim + 2), name='user_seq')
user_demo_input = Input(shape=(demo_dim,), name='user_demo')

mlp_user_embedding = Embedding(num_users + 1, embedding_dim*2, name='mlp_user_embedding')(user_id_input)
mlp_user_vec = Flatten()(mlp_user_embedding)

# sequential ratings with attention
attention_layer = Attention()(user_seq_input)
seq_features = Dense(embedding_dim, activation='relu')(attention_layer)

# demographic features
if demo_dim > 0:
    demo_features = Dense(embedding_dim//2, activation='relu')(user_demo_input)

    #all user features
    user_features = Concatenate()([mlp_user_vec, seq_features, demo_features])
else:
    user_features = Concatenate()([mlp_user_vec, seq_features])

# Final user embedding
user_final_emb = Dense(embedding_dim, activation='relu')(user_features)

user_embed_model = Model(
    inputs=[user_id_input, user_seq_input, user_demo_input],
    outputs=user_final_emb,
    name='user_embedding_model'
)

user_embeddings = user_embed_model.predict([
    train_data['user_id'],
    train_data['user_seq'],
    train_data['user_demo']
])

# Naturally color by age group
age_groups = train_ratings['age_group'].values
unique_groups = np.unique(age_groups)
age_group_map = {group: idx for idx, group in enumerate(unique_groups)}
age_group_indices = np.array([age_group_map[g] for g in age_groups])


visualize_embeddings(user_embeddings, age_group_indices, 'User')

#by gender
gender_map = {'M': 0, 'F': 1}
gender_indices = np.array([gender_map[g] for g in train_ratings['gender']])
visualize_embeddings(user_embeddings, gender_indices, 'User (by Gender)')

# by state, third cateogircal we have
state_indices = pd.Categorical(train_ratings['state']).codes
visualize_embeddings(user_embeddings, state_indices, 'User (by State)')

# %% [markdown]
# Below: in work

# %%
val_data = prepare_rnn_embedding(
    val_ratings,
    movie_items_embedded,
    embedding_dim,
    age_scaler,
    categoricals,
    user_id_map,
    movie_id_map,
    user_histories=user_histories,
    max_seq_len=max_seq_len
)
print('validation prepared')

test_data = prepare_rnn_embedding(
    test_ratings,
    movie_items_embedded,
    embedding_dim,
    age_scaler,
    categoricals,
    user_id_map,
    movie_id_map,
    user_histories=user_histories,
    max_seq_len=max_seq_len
)
print('testing prepared')

# %%
test_demo_data = prepare_rnn_embedding(
    test_ratings_with_demographics,
    movie_items_embedded,
    embedding_dim,
    age_scaler,
    categoricals,
    user_id_map,
    movie_id_map,
    user_histories=user_histories,
    max_seq_len=max_seq_len
)
print('final testing prepared')

# %%
#embed user-ratings

demo_dim = (
    1 +  #age_scaled
    len(gender_categorical_train.columns) +
    len(state_categorical_train.columns) +
    len(age_group_categorical_train.columns)
)

user_id_input = Input(shape=(1,), name='user_id')
user_seq_input = Input(shape=(max_seq_len, embedding_dim + 2), name='rating_sequence')  # ratings, movie embedding + timestamp
user_demo_input = Input(shape=(demo_dim,), name='user_demo')  # demographic vector

#embeding for each userId as fallback for cold start
user_id_emb = Embedding(input_dim=num_users + 1, output_dim=embedding_dim, mask_zero=False)(user_id_input)
user_id_emb = Flatten()(user_id_emb)

# LSTM for rating history
user_seq_emb = LSTM(128)(user_seq_input)
user_seq_emb = Dense(embedding_dim, activation='relu')(user_seq_emb)

# Combine features: RNN ratings history + demographic + uID embedding
user_full_emb = Concatenate()([user_seq_emb, user_demo_input, user_id_emb])
user_final_emb = Dense(embedding_dim, activation='relu')(user_full_emb)

user_embedding_model = Model(
    inputs=[user_id_input, user_seq_input, user_demo_input],
    outputs=user_final_emb
)

#on training input
user_embeddings = user_embedding_model.predict([
    train_data['user_id'],
    train_data['user_seq'],
    train_data['user_demo']
])

# %%
user_embeddings.shape

# %%






# %% [markdown]
# The users are definitely well embedded, there are clear regions in both PCA and UMAP

# %%

#Now do actual collaborative filter, model interactions of user-rating with movies


movie_emb_input = Input(shape=(embedding_dim,), name='movie_embedding')

#combine user embedding with movie embedding
interaction = Concatenate()([user_final_emb, movie_emb_input])
x = Dense(64, activation='relu')(interaction)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
rating_output = Dense(1, activation='linear', name='rating')(x)
# Clip values between 1-5 instead of using sigmoid scaling
rating_output = Lambda(lambda x: tf.clip_by_value(x, 1.0, 5.0))(rating_output)

# Build model
model = Model(
    inputs=[user_id_input, user_seq_input, user_demo_input, movie_emb_input],
    outputs=rating_output
)

# Compile
model.compile(optimizer='adam', loss='mse')

history = model.fit(
    x=[
        train_data['user_id'],
        train_data['user_seq'],
        train_data['user_demo'],
        train_data['movie_emb']
    ],
    y=train_data['targets'],
    validation_data=([
        val_data['user_id'],
        val_data['user_seq'],
        val_data['user_demo'],
        val_data['movie_emb']
    ], val_data['targets']),
    epochs=2,
    batch_size=32
)

# %%
#Evalu on test_data
test_predictions = model.predict([
    test_data['user_id'],
    test_data['user_seq'],
    test_data['user_demo'],
    test_data['movie_emb']
])

# %%
test_predictions.shape

# %%
test_data['targets'].shape

# %%
print("Results on test:")
evaluate_predictions(test_data['targets'], test_predictions)

plt.figure(figsize=(10, 6))
plt.scatter(test_data['targets'], test_predictions, alpha=0.5)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Test Data: Actual vs Predicted')
plt.tight_layout()
plt.show()

# %%
test_u1_predictions = model.predict([
    tf.convert_to_tensor(test_demo_data['user_id'], dtype=tf.int32),
    tf.convert_to_tensor(test_demo_data['user_seq'], dtype=tf.float32),
    tf.convert_to_tensor(test_demo_data['user_demo'], dtype=tf.float32),
    tf.convert_to_tensor(test_demo_data['movie_emb'], dtype=tf.float32)
])



# %%
print("\nResults on test_demo_data:")
evaluate_predictions(test_demo_data['targets'], test_u1_predictions)

plt.figure(figsize=(10, 6))
plt.scatter(test_demo_data['targets'], test_u1_predictions, alpha=0.5)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Test U1 Data: Actual vs Predicted')
plt.tight_layout()
plt.show()

# %%
#Return to matrix based

import implicit
import scipy.sparse as sparse

def prepare_als_data(ratings_df, movies_df, n_factors=64):
    #sparse rating matrix
    valid_movie_ids = set(movies_df['id_x'].unique()) & set(ratings_df['movieId'].unique())


    ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)]
    movies_df = movies_df[movies_df['id_x'].isin(valid_movie_ids)]

    user_ids = ratings_df['userId'].unique()
    movie_ids = movies_df['id_x'].unique()
    
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_map = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    #Map IDs to matrix indices
    row = [user_map[uid] for uid in ratings_df['userId']]
    col = [movie_map[mid] for mid in ratings_df['movieId']]
    data = ratings_df['rating'].values
    
    # Create matrix
    ratings_sparse = sparse.csr_matrix(
        (data, (row, col)), 
        shape=(len(user_ids), len(movie_ids))
    )
    
    # Get numerics and one-hot encoded features
    numeric_cols = ['revenue_norm', 'budget_norm',
       'popularity_norm', '_norm', '0_norm', '18_norm', '30_norm', '45_norm',
       'M_norm', 'M0_norm', 'M18_norm', 'M30_norm', 'M45_norm', 'F_norm',
       'F0_norm', 'F18_norm', 'F30_norm', 'F45_norm', 'release_year_norm']
    #genres
    
    # Standardize numerics
    scaler = StandardScaler()
    movie_features = movies_df[numeric_cols].copy()
    movie_features = movie_features.fillna(0)
    movie_features[numeric_cols] = scaler.fit_transform(movie_features[numeric_cols])
    
    #Add genre features
    movie_features = pd.concat([movie_features, movies_df[genres]], axis=1)
    
    #Convert to sparse
    movie_features_sparse = sparse.csr_matrix(movie_features.values)
    
    # Get relevant user features
    user_features = pd.get_dummies(ratings_df[['gender', 'state', 'age_group']]).astype('int')
    user_features['age_scaled'] = StandardScaler().fit_transform(ratings_df[['age']])
    user_features_sparse = sparse.csr_matrix(user_features.values)
    
    return (ratings_sparse, movie_features_sparse, user_features_sparse, 
            user_map, movie_map, user_ids, movie_ids)


# %%
# Train Alternating least squares
def train_als_model(ratings_sparse, movie_features_sparse, user_features_sparse, 
                    n_factors=64, regularization=0.1, alpha=40):
    
    # Initialize ALS model
    model = implicit.als.AlternatingLeastSquares(
        factors=n_factors,
        regularization=regularization,
        iterations=50
    )
    
    #convert into confidence matrix
    confidence = (ratings_sparse * alpha).astype('double')
    
    model.fit(confidence)
    
    # embeddings
    user_factors = model.user_factors
    item_factors = model.item_factors
    
    # generate biases ofr side features
    user_bias = sparse.linalg.norm(user_features_sparse[:ratings_sparse.shape[0]], axis=1).reshape(-1, 1)
    item_bias = sparse.linalg.norm(movie_features_sparse[:ratings_sparse.shape[1]], axis=1).reshape(-1, 1)
    
    # Adjust embeddings accordingly
    user_factors = user_factors * (1 + 0.1 * user_bias)
    item_factors = item_factors * (1 + 0.1 * item_bias)
    
    return model, user_factors, item_factors

# Make predictions
def predict_ratings(user_id, movie_ids, model, user_factors, item_factors,
                   user_map, movie_map):
    user_idx = user_map[user_id]
    movie_indices = [movie_map[mid] for mid in movie_ids]
    
    user_embedding = user_factors[user_idx]
    movie_embeddings = item_factors[movie_indices]
    
    # Calculate similarity by dot product for predictions
    predictions = np.dot(user_embedding, movie_embeddings.T)
    
    #scale 1-5
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 4 + 1
    
    return predictions

# %%

(ratings_sparse, movie_features_sparse, user_features_sparse,
 user_map, movie_map, user_ids, movie_ids) = prepare_als_data(ratings_with_demographics, movie_unembedded)

model, user_factors, item_factors = train_als_model(ratings_sparse, movie_features_sparse, user_features_sparse)

test_als_predictions = []
test_als_actuals = []

for _, row in test_ratings.iterrows():
    pred = predict_ratings(
        row['userId'], 
        [row['movieId']], 
        model, 
        user_factors, 
        item_factors,
        user_map, 
        movie_map
    )[0]
    test_als_predictions.append(pred)
    test_als_actuals.append(row['rating'])

#eval!
evaluate_predictions(test_als_actuals, test_als_predictions)

# %%
#to test on u1.data:

u1_als_predictions = []
u1_als_actuals = []

for _, row in test_ratings_with_demographics.iterrows():
    try:
        pred = predict_ratings(
            row['userId'], 
            [row['movieId']], 
            model, 
            user_factors, 
            item_factors,
            user_map, 
            movie_map
        )[0]
        u1_als_predictions.append(pred)
        u1_als_actuals.append(row['rating'])
    except KeyError:
        #no fallback, skip if user/movie not in training data
        continue

# Evaluate results
print("\nResults on U1.test:")
evaluate_predictions(u1_als_actuals, u1_als_predictions)

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(u1_als_actuals, u1_als_predictions, alpha=0.5)
plt.plot([1, 5], [1, 5], 'r--')  # diagonal line
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Test U1.test: ALS Actual vs Predicted')
plt.tight_layout()
plt.show()

# %%



