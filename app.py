"""Streamlit interface for the content-based movie recommender."""

from html import escape

import streamlit as st

from recommender import MovieRecommender, RECOMMENDATION_COUNT


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading movie recommendations...")
def load_recommender() -> MovieRecommender:
    return MovieRecommender.from_artifacts()


st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #101827 0%, #0b1120 420px); }
    .block-container { max-width: 1050px; padding-top: 3rem; }
    .hero { padding: 1.6rem 0 1.1rem; }
    .eyebrow { color: #f4b860; font-size: .82rem; font-weight: 700;
               letter-spacing: .14em; text-transform: uppercase; }
    .hero h1 { color: #f8fafc; font-size: clamp(2.2rem, 5vw, 3.5rem);
               margin: .4rem 0; }
    .hero p { color: #aebbd0; font-size: 1.08rem; }
    .movie-card { background: #172235; border: 1px solid #2b3b54;
                  border-radius: 14px; min-height: 120px; padding: 1.25rem; }
    .movie-number { color: #f4b860; font-size: .78rem; font-weight: 700;
                    letter-spacing: .1em; }
    .movie-title { color: #f8fafc; font-size: 1.08rem; font-weight: 650;
                   margin-top: .5rem; }
    </style>
    <div class="hero">
      <div class="eyebrow">Your next movie night</div>
      <h1>Movie Recommendation System</h1>
      <p>Content-Based Movie Recommendation using NLP and Cosine Similarity</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Pick a movie you enjoy and discover titles with similar genres, story themes, "
    "cast, and director."
)

recommender: MovieRecommender | None = None
try:
    recommender = load_recommender()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Could not load the movie recommendation data: {error}")
    st.stop()
if recommender is None:
    st.stop()

with st.form("recommendation_form"):
    selected_movie = st.selectbox(
        "Select or search for a movie",
        options=recommender.titles,
        index=None,
        placeholder="Start typing a movie title...",
    )
    submitted = st.form_submit_button(
        "Recommend movies",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not selected_movie or not selected_movie.strip():
        st.warning("Please select a movie first.")
        st.session_state.pop("recommendations", None)
    else:
        results = recommender.recommend(selected_movie, n=RECOMMENDATION_COUNT)
        if not results:
            st.info("Movie not found or no similar movies are available.")
            st.session_state.pop("recommendations", None)
        else:
            st.session_state["recommendations"] = (selected_movie, results)

if "recommendations" in st.session_state:
    selected, recommendations = st.session_state["recommendations"]
    st.divider()
    st.subheader(f"Movies similar to {selected}")
    st.caption(f"Here are up to {RECOMMENDATION_COUNT} content-based matches.")
    columns = st.columns(3)
    for index, movie in enumerate(recommendations):
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="movie-card">
                  <div class="movie-number">RECOMMENDATION {index + 1:02d}</div>
                  <div class="movie-title">{escape(movie["title"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

st.divider()
st.caption(
    "Built with Porter stemming, CountVectorizer, and cosine similarity. "
    "Recommendations use movie metadata and are not personalized to viewing history."
)
