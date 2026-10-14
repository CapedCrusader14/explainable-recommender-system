
def split_genres(value):
    """The dataset stores genres as space-separated values."""
    if not isinstance(value, str):
        return set()
    return {
        genre.strip().casefold()
        for genre in value.split()
        if genre.strip()
    }


def explain_recommendation(source_movie, recommended_movie):
    """Explain recommendations using verified shared genres."""

    source_genres = split_genres(source_movie.get("genres", ""))
    recommended_genres = split_genres(
        recommended_movie.get("genres", "")
    )

    shared_genres = sorted(source_genres & recommended_genres)

    if shared_genres:
        return "Shared genres: " + ", ".join(shared_genres)

    return (
        "No shared genres found. The recommendation was ranked "
        "using the combined TF-IDF metadata similarity."
    )
