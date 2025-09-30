import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Import dataset from file
data = pd.read_csv('data/dataset.csv', decimal=',')

# Select feature data
feature_columns = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness', 'instrumentalness', 'key', 'acousticness']
features = data[feature_columns]

# Normalize feature data to prevent feature bias
features_scaled = MinMaxScaler().fit_transform(features)

# Calculate cosine similarity
cosine_similarity = cosine_similarity(features_scaled)

def get_recommendation(track_name, cosine_sim=cosine_similarity, amount=5):
    song_index = data[data['track_name'] == track_name].index

    # Check if track exists in dataset
    if len(song_index) == 0:
        return "Track not found in the dataset."

    # Get the first index from list
    song_index = song_index[0]

    # Get similarity scores
    similarity_socres = list(enumerate(cosine_sim[song_index]))
    similarity_socres = sorted(similarity_socres, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar tracks
    similarity_socres = similarity_socres[1:amount+1]
    track_indices = [i[0] for i in similarity_socres]

    return data[['track_name', 'track_artist']].iloc[track_indices]


print(get_recommendation('Good Luck, Babe!'))