from flask import Flask, request, jsonify, render_template, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import random

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    liked_movies = db.Column(db.String, default='')  # Change to string
    disliked_movies = db.Column(db.String, default='')  # Change to string
    recent_liked_genre = db.Column(db.String, default='')
    
# Create the database
with app.app_context():
    db.create_all()

# Registration route
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400

    user = User.query.filter_by(username=username).first()
    if user:
        return jsonify({'message': 'Username already exists!'}), 400

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully!'}), 201

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400

    user = User.query.filter_by(username=username).first()

    if user and user.password == password:
        # Send a response with a cookie containing the username
        response = make_response(jsonify({'message': 'Login successful!'}), 200)
        response.set_cookie('username', username)
        return response
    else:
        return jsonify({'message': 'Invalid username or password!'}), 401



# Logout route
@app.route('/logout', methods=['POST'])
def logout():
    return jsonify({'message': 'Logout successful!'}), 200

# Load the dataset
movies_data = pd.read_csv('movies_dataset.csv')

# Step 1: Prioritize movies by likes, filter by genres, and include genre details
def get_top_liked_movies_by_genres(movies_data, genres, num_top=15):
    genre_filtered = movies_data[movies_data['genre'].apply(lambda x: any(genre.lower() in x.lower() for genre in genres))]
    genre_filtered = genre_filtered[genre_filtered['likes'] >= genre_filtered['dislikes']]
    sorted_movies = genre_filtered.sort_values(by='likes', ascending=False)
    return sorted_movies[['id', 'movie_name', 'likes', 'dislikes', 'genre']].head(num_top)  # Return ID

# Step 2: Use KNN to recommend similar movies
def recommend_similar_movies(top_liked_movies, num_recommendations=15, seed=None):
    # Load the dataset every time this function is called
    movies_data = pd.read_csv('movies_dataset.csv')

    # Use KNN for recommendations
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['combined_features'])
    svd = TruncatedSVD(n_components=50)
    svd_matrix = svd.fit_transform(tfidf_matrix)

    # Initialize KNN model
    knn_model = NearestNeighbors(n_neighbors=num_recommendations + 1, algorithm='brute', metric='cosine')
    knn_model.fit(svd_matrix)

    # Get indices of top liked movies
    top_liked_indices = top_liked_movies.index.tolist()

    recommendations = []
    for index in top_liked_indices:
        distances, indices = knn_model.kneighbors(svd_matrix[index].reshape(1, -1), n_neighbors=num_recommendations + 1)
        recommended_indices = indices.flatten()[1:]  # Skip the first index (itself)

        if seed is not None:
            random.seed(seed)
        random.shuffle(recommended_indices)

        for i in recommended_indices:
            recommended_movie = movies_data.iloc[i]
            if recommended_movie['likes'] >= recommended_movie['dislikes']:
                recommendations.append(recommended_movie)

    # Combine top liked movies and recommendations
    recommendations_df = pd.DataFrame(recommendations).drop_duplicates(subset='movie_name')
    
    # Combine the top liked movies and the recommended ones
    recommendations_combined = pd.concat([top_liked_movies, recommendations_df]).drop_duplicates(subset='movie_name')
    
    # Sort by likes and limit to num_recommendations
    recommendations_combined = recommendations_combined.sort_values(by='likes', ascending=False).head(num_recommendations)

    return recommendations_combined[['id', 'movie_name', 'likes', 'dislikes', 'genre']]  # Return movie ID

# # Function to update movie likes and dislikes based on movie ID
# def update_movie_likes_by_id(movie_id, username, action_type):
#     user = User.query.filter_by(username=username).first()
#     print(str(movie_id) + " " + str(username) + " " + str(action_type))
    
#     if not user:
#         return {"message": "User not found"}, 404

#     # Load the dataset
#     try:
#         movies_data = pd.read_csv('movies_dataset.csv')
#         movie = movies_data[movies_data['id'] == movie_id]
#     except Exception as e:
#         return {"message": "Error reading movie data"}, 500

#     if movie.empty:
#         return {"message": "Movie not found"}, 404

#     # Get the current movie's likes and dislikes
#     current_likes = movie['likes'].values[0]
#     current_dislikes = movie['dislikes'].values[0]
#     genre = movie['genre'].values[0]  # Get the genre of the movie

#     if action_type == 'like':
#         if str(movie_id) in user.liked_movies.split(','):
#             return {"message": "You have already liked this movie"}, 400  # Prevent multiple likes
#         current_likes += 1
#         user.liked_movies += (',' + str(movie_id)) if user.liked_movies else str(movie_id)  # Append movie ID
#         user.recent_liked_genre = genre

#     elif action_type == 'dislike':
#         if str(movie_id) in user.disliked_movies.split(','):
#             return {"message": "You have already disliked this movie"}, 400
#         current_dislikes += 1
#         user.disliked_movies += (',' + str(movie_id)) if user.disliked_movies else str(movie_id)  # Append movie ID

#     # Update the user's movie likes/dislikes in the database
#     db.session.commit()

#     # Update the CSV file with the new likes/dislikes
#     movies_data.loc[movies_data['id'] == movie_id, 'likes'] = current_likes
#     movies_data.loc[movies_data['id'] == movie_id, 'dislikes'] = current_dislikes
#     movies_data.to_csv('movies_dataset.csv', index=False)

#     return {"message": f"Movie {action_type}d successfully"}, 200

def update_movie_likes_by_id(movie_id, username, action_type):
    user = User.query.filter_by(username=username).first()
    
    if not user:
        return {"message": "User not found"}, 404

    # Load the dataset
    try:
        movies_data = pd.read_csv('movies_dataset.csv')
        movie = movies_data[movies_data['id'] == movie_id]
    except Exception as e:
        return {"message": "Error reading movie data"}, 500

    if movie.empty:
        return {"message": "Movie not found"}, 404

    # Get the current movie's likes and dislikes
    current_likes = movie['likes'].values[0]
    current_dislikes = movie['dislikes'].values[0]
    genre = movie['genre'].values[0]  # Get the genre of the movie

    liked_movies = user.liked_movies.split(',') if user.liked_movies else []
    disliked_movies = user.disliked_movies.split(',') if user.disliked_movies else []

    if action_type == 'like':
        if str(movie_id) in liked_movies:
            return {"message": "You have already liked this movie"}, 400  # Prevent multiple likes
        elif str(movie_id) in disliked_movies:
            # If user had disliked it before, remove from disliked, decrease dislike count, increase like count
            current_dislikes -= 1
            disliked_movies.remove(str(movie_id))

        # Add to liked movies and increase like count
        liked_movies.append(str(movie_id))
        current_likes += 1
        user.liked_movies = ','.join(liked_movies)
        user.disliked_movies = ','.join(disliked_movies)
        user.recent_liked_genre = genre

    elif action_type == 'dislike':
        if str(movie_id) in disliked_movies:
            return {"message": "You have already disliked this movie"}, 400
        elif str(movie_id) in liked_movies:
            # If user had liked it before, remove from liked, decrease like count, increase dislike count
            current_likes -= 1
            liked_movies.remove(str(movie_id))

        # Add to disliked movies and increase dislike count
        disliked_movies.append(str(movie_id))
        current_dislikes += 1
        user.liked_movies = ','.join(liked_movies)
        user.disliked_movies = ','.join(disliked_movies)

    # Commit the changes to the user's likes/dislikes in the database
    db.session.commit()

    # Update the CSV file with the new like/dislike counts
    movies_data.loc[movies_data['id'] == movie_id, 'likes'] = current_likes
    movies_data.loc[movies_data['id'] == movie_id, 'dislikes'] = current_dislikes
    movies_data.to_csv('movies_dataset.csv', index=False)

    return {"message": f"Movie {action_type}d successfully"}, 200


@app.route('/movie/like', methods=['POST'])
def like_movie():
    data = request.get_json()
    movie_id = data.get('movie_id')  # Expecting movie ID now
    username = data.get('username')
    
    try:
        movie_id = int(movie_id)  # Ensure movie_id is an integer
    except (TypeError, ValueError):
        return jsonify({"message": "Invalid movie ID provided"}), 400
    
    return update_movie_likes_by_id(movie_id, username, 'like')

@app.route('/movie/dislike', methods=['POST'])
def dislike_movie():
    data = request.get_json()
    movie_id = data.get('movie_id')  # Expecting movie ID now
    username = data.get('username')
    
    try:
        movie_id = int(movie_id)  # Ensure movie_id is an integer
    except (TypeError, ValueError):
        return jsonify({"message": "Invalid movie ID provided"}), 400
    
    return update_movie_likes_by_id(movie_id, username, 'dislike')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    input_genres = data.get('genres', [])
    top_liked_movies = get_top_liked_movies_by_genres(movies_data, input_genres, num_top=15)
    
    # Now the recommendations will include movie IDs
    recommendations = top_liked_movies.to_dict(orient='records')
    return jsonify({'recommendations': recommendations})


# Route to get the recently liked genre of a user
@app.route('/recent-liked-genre', methods=['POST'])
def recent_liked_genre():
    data = request.get_json()
    username = data.get('username')
    
    if not username:
        return jsonify({'message': 'Username is required'}), 400

    # Fetch the user
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Ensure recent_liked_genre exists and is formatted correctly
    recent_genre_str = user.recent_liked_genre
    if not recent_genre_str:
        return jsonify({'message': 'No genres liked yet.'}), 200
    
    # Parse the recent_liked_genre as a list
    input_genres = [genre.strip() for genre in recent_genre_str.split(',') if genre.strip()]
    if not input_genres:
        return jsonify({'message': 'No valid genres found in recent likes.'}), 200

    # Get top liked movies by genre
    try:
        top_liked_movies = get_top_liked_movies_by_genres(movies_data, input_genres, num_top=15)
    except Exception as e:
        return jsonify({'message': 'Error retrieving movies by genre', 'error': str(e)}), 500

    # Convert recommendations to dictionary format
    recommendations = top_liked_movies.to_dict(orient='records')
    return jsonify({'recommendations': recommendations})



# Routes for rendering HTML templates
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/loginpage')
def loginpage():
    return render_template('login.html')

@app.route('/registerpage')
def registerpage():
    return render_template('register.html')

@app.route('/recommendpage')
def recommendpage():
    return render_template('recommend.html')

@app.route('/trendingpage')
def trendingpage():
    return render_template('trending.html')

@app.route('/mymoviespage')
def mymoviespage():
    return render_template('mymovies.html')

if __name__ == '__main__':
    app.run(debug=True)
