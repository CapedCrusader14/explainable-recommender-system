
from recommender import Recommender
from explanations import explain_recommendation

model = Recommender()

movie_id = 1
source = model.movies.loc[
    model.movies["movie_id"] == movie_id
].iloc[0]

recommendations = model.recommend(movie_id, top_n=5)

print(f"\nRecommendations similar to: {source['title']}\n")

for _, recommendation in recommendations.iterrows():
    target = model.movies.loc[
        model.movies["movie_id"] == recommendation["movie_id"]
    ].iloc[0]

    explanation = explain_recommendation(source, target)

    print(f"Movie: {recommendation['title']}")
    print(f"Similarity score: {recommendation['similarity']}")
    print(f"Explanation: {explanation}")
    print("-" * 60)
