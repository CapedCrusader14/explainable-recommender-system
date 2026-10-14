
from recommender import Recommender
from explanations import explain_recommendation

model = Recommender()
movies = model.movies

print("\n===== EXPLAINABLE MOVIE RECOMMENDER =====")
print(f"Available movies: {len(movies)}")
print("Type part of a movie title to search.")
print("Type 'exit' to quit.\n")

while True:
    query = input("Enter a movie title: ").strip()

    if query.lower() == "exit":
        print("Thank you for using the recommender!")
        break

    if not query:
        print("Please enter a movie title.\n")
        continue

    # Find movies containing the entered text (case-insensitive)
    matches = movies[
        movies["title"].str.contains(
            query, case=False, na=False, regex=False
        )
    ]

    if matches.empty:
        print("No matching movies found. Try another title.\n")
        continue

    # Display choices if multiple titles match
    print("\nMatching movies:")
    for _, row in matches.head(20).iterrows():
        print(f"ID: {row['movie_id']} | {row['title']} ({row['year']})")

    try:
        movie_id = int(input("\nEnter the ID of the movie you want: "))
    except ValueError:
        print("Please enter a valid numeric movie ID.\n")
        continue

    selected = movies[movies["movie_id"] == movie_id]

    if selected.empty or movie_id not in matches["movie_id"].values:
        print("Choose an ID from the matching movies above.\n")
        continue

    source = selected.iloc[0]

    try:
        top_n = int(input("How many recommendations? (1-20): "))
        if not 1 <= top_n <= 20:
            print("Please choose a number between 1 and 20.\n")
            continue
    except ValueError:
        print("Please enter a valid number.\n")
        continue

    recommendations = model.recommend(movie_id, top_n=top_n)

    print(f"\nRecommendations similar to: {source['title']}")
    print("=" * 60)

    for _, rec in recommendations.iterrows():
        target = movies[movies["movie_id"] == rec["movie_id"]].iloc[0]
        explanation = explain_recommendation(source, target)

        print(f"\nMovie: {rec['title']} ({rec['year']})")
        print(f"Similarity score: {rec['similarity']:.3f}")
        print(f"Genres: {rec['genres'] or 'Not available'}")
        print(f"Explanation: {explanation}")

    print("\n" + "=" * 60 + "\n")
