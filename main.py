import pandas as pd
from fastapi import FastAPI, Response, Request, Form
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from starlette.responses import HTMLResponse

from models.Similarity import Similarity

# Import dataset from file
data = pd.read_csv('data/dataset.csv', decimal=',')

# Select feature data
feature_columns = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness', 'instrumentalness', 'key', 'acousticness']
categorical_columns = ['playlist_genre', 'playlist_subgenre']

# Get only categorical columns from dataset
categorical_data = data[categorical_columns]

# One-hot encode categorical features
categorical_features = pd.get_dummies(categorical_data, columns=categorical_columns)

features = data[feature_columns]

# Normalize feature data to prevent feature bias
features_scaled = MinMaxScaler().fit_transform(features)

# Combine numerical and categorical features
complete_features = pd.concat([pd.DataFrame(features_scaled), categorical_features.reset_index(drop=True)], axis=1)

def get_cosine_similarity():
    # Calculate cosine similarity
    return cosine_similarity(complete_features)


def get_euclidean_similarity():
    # Calculate euclidean distance
    euclidean_dist = euclidean_distances(complete_features)
    # Calculate euclidean similarity score
    euclidean_similarity = 1 / (1 + euclidean_dist)
    return euclidean_similarity

def get_recommendation(song_id, similarity=get_cosine_similarity(), amount=5):
    song_index = data[data['song_id'] == song_id].index

    # Check if track exists in dataset
    if len(song_index) == 0:
        return "Track not found in the dataset."

    # Get the first index from list
    song_index = song_index[0]

    # Get similarity scores
    similarity_scores = list(enumerate(similarity[song_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar tracks
    similarity_scores = similarity_scores[1:amount+1]
    track_indices = [i[0] for i in similarity_scores]
    scores = [i[1] for i in similarity_scores]

    result = data[['track_name', 'track_artist']].iloc[track_indices].copy()
    result['similarity_score'] = scores

    return result


app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name = "index.html")


@app.post("/recommend", status_code=200)
def show_recommendation(song_id: int = Form(...), amount: int = Form(...), similarity: int = Form(...), request: Request = None):
    if similarity == Similarity.Cosine.value:
        recommendations = get_recommendation(song_id, similarity=get_cosine_similarity(), amount=amount)
        return templates.TemplateResponse(request=request, name="result.html", context={"recommendations": recommendations.to_dict('records')})
    elif similarity == Similarity.Euclidean.value:
        recommendations = get_recommendation(song_id, similarity=get_euclidean_similarity(), amount=amount)
        return templates.TemplateResponse(request=request, name="result.html", context={"recommendations": recommendations.to_dict('records')})
    else:
        return Response(status_code=400, content="Invalid similarity type")


@app.get("/search")
def search_songs(query: str):
    if not query or len(query) < 2:
        return []

    matches = data[data['track_name'].str.contains(query, case=False, na=False)]
    matches = matches.head(10)
    results = matches[['song_id', 'track_name', 'track_artist']].to_dict('records')

    return results



#print(get_recommendation('Good Luck, Babe!', get_cosine_similarity(), 20))