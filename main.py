import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#1D1D20"
TEXT     = "#fbfbff"
SUBTEXT  = "#909094"
GOLD     = "#ffd400"
GREEN    = "#17b26a"
RED      = "#f04438"
PALETTE  = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF", "#ffd400"]
GENRE_COLORS = [
    "#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
    "#F7B6D2","#1F77B4","#9467BD","#8C564B","#C49C94",
    "#E377C2","#ffd400"
]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": SUBTEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": "#333338",
    "font.family": "sans-serif",
})

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Intelligence Lab",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* Global dark theme */
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ background-color: #141416; }}
  .stTabs [data-baseweb="tab-list"] {{ background-color: #141416; border-radius: 8px; padding: 4px; }}
  .stTabs [data-baseweb="tab"] {{ color: {SUBTEXT}; background-color: transparent; border-radius: 6px; }}
  .stTabs [aria-selected="true"] {{ background-color: #2a2a30 !important; color: {TEXT} !important; }}
  .metric-card {{
      background: #23232a; border-radius: 12px; padding: 20px 24px;
      border: 1px solid #333340; margin-bottom: 12px;
  }}
  .metric-value {{ font-size: 2.8rem; font-weight: 800; color: {GOLD}; }}
  .metric-label {{ font-size: 0.85rem; color: {SUBTEXT}; text-transform: uppercase; letter-spacing: 1px; }}
  .mood-card {{
      background: linear-gradient(135deg, #23232a, #1a1a22);
      border-radius: 16px; padding: 28px; text-align: center;
      border: 2px solid {GOLD}; margin: 16px 0;
  }}
  .mood-emoji {{ font-size: 4rem; }}
  .mood-name {{ font-size: 1.6rem; font-weight: 700; color: {GOLD}; margin: 8px 0; }}
  .mood-desc {{ color: {SUBTEXT}; font-size: 0.95rem; line-height: 1.6; }}
  h1, h2, h3 {{ color: {TEXT} !important; }}
  .stSlider > div > div > div > div {{ background: {GOLD}; }}
  hr {{ border-color: #333340; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA + MODEL TRAINING (lightweight, cached)
# ─────────────────────────────────────────────────────────────────────────────
GENRES = [
    "Pop", "Hip-Hop", "Rock", "Electronic/Dance", "R&B/Soul",
    "Country", "Latin", "Folk/Acoustic", "Jazz", "Classical", "Metal", "Indie"
]
DECADES = [1960, 1970, 1980, 1990, 2000, 2010, 2020]

GENRE_PARAMS = {
    "Pop":              {"tempo":(118,14), "energy":(0.73,0.13), "dance":(0.72,0.12), "valence":(0.60,0.17), "acoustic":(0.11,0.12), "loud":(-5.2,2.4), "speech":(0.06,0.04), "instrum":(0.02,0.05)},
    "Hip-Hop":          {"tempo":(96,15),  "energy":(0.72,0.14), "dance":(0.78,0.11), "valence":(0.52,0.19), "acoustic":(0.09,0.10), "loud":(-5.8,2.6), "speech":(0.19,0.09), "instrum":(0.04,0.07)},
    "Rock":             {"tempo":(128,18), "energy":(0.82,0.12), "dance":(0.52,0.14), "valence":(0.47,0.18), "acoustic":(0.10,0.11), "loud":(-5.0,2.8), "speech":(0.07,0.04), "instrum":(0.07,0.10)},
    "Electronic/Dance": {"tempo":(128,10), "energy":(0.80,0.12), "dance":(0.77,0.11), "valence":(0.55,0.18), "acoustic":(0.04,0.06), "loud":(-5.3,2.5), "speech":(0.07,0.04), "instrum":(0.18,0.20)},
    "R&B/Soul":         {"tempo":(100,16), "energy":(0.63,0.15), "dance":(0.73,0.13), "valence":(0.52,0.18), "acoustic":(0.15,0.14), "loud":(-6.4,2.7), "speech":(0.10,0.05), "instrum":(0.04,0.08)},
    "Country":          {"tempo":(119,16), "energy":(0.64,0.15), "dance":(0.60,0.14), "valence":(0.57,0.19), "acoustic":(0.29,0.20), "loud":(-6.8,2.8), "speech":(0.05,0.03), "instrum":(0.04,0.09)},
    "Latin":            {"tempo":(110,17), "energy":(0.73,0.13), "dance":(0.77,0.11), "valence":(0.65,0.16), "acoustic":(0.14,0.14), "loud":(-5.9,2.5), "speech":(0.10,0.05), "instrum":(0.05,0.09)},
    "Folk/Acoustic":    {"tempo":(112,17), "energy":(0.47,0.15), "dance":(0.54,0.15), "valence":(0.50,0.19), "acoustic":(0.47,0.22), "loud":(-8.9,3.0), "speech":(0.05,0.03), "instrum":(0.08,0.15)},
    "Jazz":             {"tempo":(120,22), "energy":(0.54,0.16), "dance":(0.55,0.15), "valence":(0.53,0.19), "acoustic":(0.33,0.22), "loud":(-9.0,3.2), "speech":(0.06,0.04), "instrum":(0.24,0.28)},
    "Classical":        {"tempo":(115,24), "energy":(0.31,0.15), "dance":(0.34,0.14), "valence":(0.36,0.17), "acoustic":(0.76,0.18), "loud":(-14.5,4.5), "speech":(0.04,0.02), "instrum":(0.78,0.22)},
    "Metal":            {"tempo":(140,20), "energy":(0.92,0.07), "dance":(0.41,0.13), "valence":(0.35,0.16), "acoustic":(0.05,0.07), "loud":(-4.5,2.2), "speech":(0.07,0.04), "instrum":(0.09,0.14)},
    "Indie":            {"tempo":(115,18), "energy":(0.61,0.16), "dance":(0.57,0.15), "valence":(0.48,0.18), "acoustic":(0.24,0.19), "loud":(-7.8,3.2), "speech":(0.06,0.04), "instrum":(0.09,0.14)},
}

DECADE_WEIGHTS = np.array([0.04, 0.06, 0.10, 0.16, 0.20, 0.24, 0.20])
DECADE_ACOUSTIC_MEANS = {1960: 0.31, 1970: 0.22, 1980: 0.15, 1990: 0.12, 2000: 0.09, 2010: 0.06, 2020: 0.04}
DECADE_LOUD_MEANS = {1960: -7.6, 1970: -7.0, 1980: -6.5, 1990: -5.8, 2000: -5.2, 2010: -4.8, 2020: -3.5}
DECADE_ENERGY_MEANS = {1960: 0.63, 1970: 0.66, 1980: 0.70, 1990: 0.71, 2000: 0.73, 2010: 0.76, 2020: 0.78}

MOOD_DESCRIPTIONS = {
    "🔥 Euphoric Dance":       "High energy, highly danceable, positive vibes. Built for the dancefloor. Think EDM bangers and festival anthems.",
    "🎤 High-Energy Hip-Hop":  "Energetic beats with rhythmic speech patterns. Dominates streaming charts. Raw, confident, and expressive.",
    "⚡ Aggressive Rock":       "Maximum energy, heavy and loud. Low acousticness, powerful guitar riffs. Not for the faint-hearted.",
    "🌧 Melancholic Indie":    "Acoustic-leaning, emotionally complex. Lower energy but deeply resonant. Perfect for introspective moods.",
    "🎸 Chill Acoustic":       "Warm acoustic tones, moderate tempo. The sound of Sunday mornings and coffee shops.",
    "🎹 Ambient / Instrumental":"Instrumental-heavy with atmospheric qualities. Music that lets you breathe. Great for focus and study.",
}

@st.cache_resource
def build_model_and_data():
    """Generate synthetic dataset and train lightweight GBM model."""
    rng = np.random.RandomState(42)
    N = 5000
    rows = []
    genres_arr = rng.choice(GENRES, N, p=[0.15,0.12,0.12,0.10,0.09,0.08,0.08,0.07,0.05,0.05,0.05,0.04])
    decade_arr = rng.choice(DECADES, N, p=DECADE_WEIGHTS)

    for g, d in zip(genres_arr, decade_arr):
        p = GENRE_PARAMS[g]
        decade_norm = (d - 1960) / 60.0
        t = float(np.clip(rng.normal(p["tempo"][0], p["tempo"][1]), 60, 200))
        e = float(np.clip(rng.normal(p["energy"][0] + 0.12 * decade_norm, p["energy"][1]), 0, 1))
        da = float(np.clip(rng.normal(p["dance"][0] + 0.05 * decade_norm, p["dance"][1]), 0, 1))
        v = float(np.clip(rng.normal(p["valence"][0], p["valence"][1]), 0, 1))
        ac = float(np.clip(rng.normal(p["acoustic"][0] - 0.20 * decade_norm, p["acoustic"][1]), 0, 1))
        ld = float(np.clip(rng.normal(p["loud"][0] + 3.0 * decade_norm, p["loud"][1]), -30, 0))
        sp = float(np.clip(rng.normal(p["speech"][0], p["speech"][1]), 0, 1))
        ins = float(np.clip(rng.normal(p["instrum"][0], p["instrum"][1]), 0, 1))
        yr = int(rng.randint(d, min(d + 9, 2024)))
        dur = int(rng.normal(210, 35))
        mode = rng.choice(["major", "minor"], p=[0.60, 0.40])
        # Popularity formula
        pop_raw = (45 + e * 25 + da * 22 + v * 10 + (ld + 10) * 1.5
                   + 8 * (yr - 1960) / 64 + rng.normal(0, 6))
        pop = int(np.clip(pop_raw, 0, 100))
        rows.append([yr, d, g, t, e, da, v, ac, ld, sp, ins, dur, mode, pop])

    df = pd.DataFrame(rows, columns=[
        "year","decade","genre","tempo","energy","danceability","valence",
        "acousticness","loudness_db","speechiness","instrumentalness",
        "duration_sec","mode","popularity"
    ])

    le_g = LabelEncoder().fit(GENRES)
    le_m = LabelEncoder().fit(["major", "minor"])
    df["genre_enc"] = le_g.transform(df["genre"])
    df["mode_enc"]  = le_m.transform(df["mode"])

    FEATS = ["tempo","energy","danceability","valence","acousticness",
             "loudness_db","speechiness","instrumentalness","duration_sec",
             "year","genre_enc","mode_enc"]
    X_tr = df[FEATS].values
    y_tr = df["popularity"].values.astype(float)

    gbm = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.10,
        subsample=0.8, min_samples_leaf=15, max_features=0.8, random_state=42
    )
    gbm.fit(X_tr, y_tr)

    # KMeans mood clustering
    CLUSTER_FEATS = ["tempo","energy","danceability","valence",
                     "acousticness","loudness_db","speechiness","instrumentalness"]
    scaler = StandardScaler().fit(df[CLUSTER_FEATS])
    X_sc = scaler.transform(df[CLUSTER_FEATS])
    km = KMeans(n_clusters=6, random_state=42, n_init=20)
    km.fit(X_sc)

    # Name clusters by centroid rules
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    c_df = pd.DataFrame(centroids_orig, columns=CLUSTER_FEATS)

    def name_c(row):
        if row["energy"] > 0.78 and row["danceability"] > 0.65 and row["valence"] > 0.55:
            return "🔥 Euphoric Dance"
        elif row["energy"] > 0.70 and (row["speechiness"] > 0.12 or row["danceability"] > 0.60) and row["valence"] > 0.40:
            return "🎤 High-Energy Hip-Hop"
        elif row["energy"] > 0.72 and row["loudness_db"] > -7 and row["acousticness"] < 0.15 and row["danceability"] < 0.55:
            return "⚡ Aggressive Rock"
        elif row["acousticness"] > 0.35 and row["energy"] < 0.60 and row["valence"] < 0.52:
            return "🌧 Melancholic Indie"
        elif row["acousticness"] > 0.25 and row["energy"] < 0.65 and row["valence"] > 0.48 and row["tempo"] < 120:
            return "🎸 Chill Acoustic"
        else:
            return "🎹 Ambient / Instrumental"

    cluster_map = {i: name_c(c_df.loc[i]) for i in range(6)}

    return gbm, le_g, le_m, scaler, km, cluster_map, FEATS, CLUSTER_FEATS, df


gbm_model, le_genre, le_mode, cl_scaler, kmeans_model, cluster_map, MODEL_FEATS, CLUSTER_FEATS, songs_data = build_model_and_data()


def predict_popularity(tempo, energy, danceability, valence, acousticness,
                       loudness_db, speechiness, instrumentalness,
                       duration_sec, year, genre, mode):
    genre_enc = int(le_genre.transform([genre])[0])
    mode_enc  = int(le_mode.transform([mode])[0])
    row = np.array([[tempo, energy, danceability, valence, acousticness,
                     loudness_db, speechiness, instrumentalness,
                     duration_sec, year, genre_enc, mode_enc]])
    pred = float(gbm_model.predict(row)[0])
    return round(np.clip(pred, 0, 100), 1)


def predict_mood(tempo, energy, danceability, valence, acousticness,
                 loudness_db, speechiness, instrumentalness):
    row = np.array([[tempo, energy, danceability, valence, acousticness,
                     loudness_db, speechiness, instrumentalness]])
    row_sc = cl_scaler.transform(row)
    cluster_id = int(kmeans_model.predict(row_sc)[0])
    return cluster_map[cluster_id]


def decade_chart_score(tempo, energy, danceability, valence, acousticness,
                       loudness_db, speechiness, instrumentalness, decade):
    """Estimate how well a song's audio profile fits a given decade."""
    energy_fit     = 1 - abs(energy - DECADE_ENERGY_MEANS[decade]) * 2
    acoustic_fit   = 1 - abs(acousticness - DECADE_ACOUSTIC_MEANS[decade]) * 2
    loudness_fit   = 1 - abs(loudness_db - DECADE_LOUD_MEANS[decade]) / 10
    base_pop = predict_popularity(tempo, energy, danceability, valence, acousticness,
                                  loudness_db, speechiness, instrumentalness,
                                  210, decade + 5, "Pop", "major")
    decade_fit = (energy_fit + acoustic_fit + loudness_fit) / 3
    score = base_pop * (0.65 + 0.35 * decade_fit)
    return round(np.clip(score, 5, 98), 1)


def recommend_genre(tempo, energy, danceability, valence, acousticness,
                    loudness_db, speechiness, instrumentalness):
    """Score each genre by profile similarity using Euclidean distance."""
    scores = {}
    for g, p in GENRE_PARAMS.items():
        g_vec = np.array([
            p["tempo"][0] / 200,
            p["energy"][0],
            p["dance"][0],
            p["valence"][0],
            p["acoustic"][0],
            (p["loud"][0] + 30) / 30,
            p["speech"][0],
            p["instrum"][0],
        ])
        u_vec = np.array([
            tempo / 200,
            energy,
            danceability,
            valence,
            acousticness,
            (loudness_db + 30) / 30,
            speechiness,
            instrumentalness,
        ])
        dist = np.linalg.norm(g_vec - u_vec)
        scores[g] = dist

    # Convert distance to similarity %
    sorted_genres = sorted(scores.items(), key=lambda x: x[1])
    max_dist = max(d for _, d in sorted_genres)
    ranked = [(g, round((1 - d / max_dist) * 100, 1)) for g, d in sorted_genres]
    return ranked  # [(genre, similarity_pct), ...]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 24px 0 8px 0;">
  <h1 style="font-size:2.5rem; font-weight:900; margin:0;">
    🎵 Music Intelligence Lab
  </h1>
  <p style="color:#909094; font-size:1.05rem; margin-top:8px;">
    Dial in your song's audio DNA and discover its commercial potential, mood archetype, decade fit, and genre identity.
  </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — AUDIO CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 16px 0 8px 0;">
      <div style="font-size:2.5rem;">🎛️</div>
      <div style="font-size:1.2rem; font-weight:700; color:{TEXT};">Audio DNA Controls</div>
      <div style="font-size:0.8rem; color:{SUBTEXT};">Adjust sliders to build your song's profile</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    tempo         = st.slider("🥁 Tempo (BPM)",        60,  200,  120, step=1)
    energy        = st.slider("⚡ Energy",               0.0, 1.0, 0.70, step=0.01)
    danceability  = st.slider("💃 Danceability",         0.0, 1.0, 0.65, step=0.01)
    valence       = st.slider("😊 Valence (Positivity)", 0.0, 1.0, 0.55, step=0.01)
    acousticness  = st.slider("🎸 Acousticness",         0.0, 1.0, 0.15, step=0.01)

    st.divider()
    st.markdown(f"<div style='color:{SUBTEXT}; font-size:0.8rem; margin-bottom:4px;'>ADVANCED</div>", unsafe_allow_html=True)
    loudness_db   = st.slider("🔊 Loudness (dBFS)",    -25.0, 0.0, -6.0, step=0.5)
    speechiness   = st.slider("🎤 Speechiness",          0.0, 1.0, 0.07, step=0.01)
    instrumentalness = st.slider("🎹 Instrumentalness",  0.0, 1.0, 0.05, step=0.01)
    duration_sec  = st.slider("⏱ Duration (seconds)",   60,  420,  210, step=5)

    st.divider()
    st.markdown(f"<div style='color:{SUBTEXT}; font-size:0.8rem; margin-bottom:4px;'>METADATA</div>", unsafe_allow_html=True)
    year          = st.slider("📅 Release Year",        1960, 2024, 2022, step=1)
    genre_choice  = st.selectbox("🎼 Genre", GENRES, index=0)
    mode_choice   = st.selectbox("🎵 Mode", ["major", "minor"], index=0)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE ALL PREDICTIONS (run once per slider change)
# ─────────────────────────────────────────────────────────────────────────────
pop_score   = predict_popularity(tempo, energy, danceability, valence, acousticness,
                                 loudness_db, speechiness, instrumentalness,
                                 duration_sec, year, genre_choice, mode_choice)
mood_name   = predict_mood(tempo, energy, danceability, valence, acousticness,
                           loudness_db, speechiness, instrumentalness)
genre_ranks = recommend_genre(tempo, energy, danceability, valence, acousticness,
                              loudness_db, speechiness, instrumentalness)
decade_scores = {d: decade_chart_score(tempo, energy, danceability, valence, acousticness,
                                       loudness_db, speechiness, instrumentalness, d)
                 for d in DECADES}

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
pop_pct = pop_score
if pop_pct >= 75:
    pop_label, pop_color = "🔥 CHART HIT", GOLD
elif pop_pct >= 55:
    pop_label, pop_color = "📈 SOLID TRACK", GREEN
elif pop_pct >= 35:
    pop_label, pop_color = "📻 RADIO POTENTIAL", "#A1C9F4"
else:
    pop_label, pop_color = "🎭 NICHE APPEAL", SUBTEXT

best_decade = max(decade_scores, key=decade_scores.get)
best_genre  = genre_ranks[0][0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Popularity Score</div>
      <div class="metric-value" style="color:{pop_color};">{pop_score}</div>
      <div style="color:{pop_color}; font-weight:600; font-size:0.9rem;">{pop_label}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Mood Archetype</div>
      <div style="font-size:1.4rem; font-weight:700; color:{GOLD}; margin:6px 0;">{mood_name}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Best Decade Fit</div>
      <div class="metric-value">{best_decade}s</div>
      <div style="color:{SUBTEXT}; font-size:0.85rem;">Score: {decade_scores[best_decade]}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Top Genre Match</div>
      <div style="font-size:1.3rem; font-weight:700; color:{GOLD}; margin:6px 0;">{best_genre}</div>
      <div style="color:{SUBTEXT}; font-size:0.85rem;">Similarity: {genre_ranks[0][1]}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Popularity Predictor",
    "🧬 Mood Archetype",
    "⏳ Decade Time Machine",
    "🎼 Genre Recommender"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — POPULARITY PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔮 Real-Time Popularity Prediction")
    st.markdown(f"<p style='color:{SUBTEXT};'>Our Gradient Boosting model (R²≈0.73) predicts where your track lands on the 0–100 popularity scale based on its audio DNA.</p>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        # Gauge-style chart
        fig_gauge, ax_g = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=False))
        fig_gauge.patch.set_facecolor(BG)
        ax_g.set_facecolor(BG)

        # Background bar
        ax_g.barh([""], [100], color="#2a2a30", height=0.5, edgecolor="none")
        # Score bar
        bar_color = pop_color if pop_pct >= 55 else "#A1C9F4"
        ax_g.barh([""], [pop_score], color=bar_color, height=0.5, edgecolor="none")
        ax_g.set_xlim(0, 100)
        ax_g.set_xlabel("Popularity Score (0–100)", color=SUBTEXT, fontsize=11)
        ax_g.set_title(f"Predicted Popularity: {pop_score}", color=TEXT, fontsize=16, fontweight="bold", pad=14)
        ax_g.text(pop_score + 1, 0, f"{pop_score}", va="center", color=TEXT, fontsize=14, fontweight="bold")

        # Zone markers
        for threshold, label, color in [(35, "Niche", SUBTEXT), (55, "Radio", "#A1C9F4"), (75, "Hit", GOLD)]:
            ax_g.axvline(threshold, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
            ax_g.text(threshold, 0.32, label, color=color, fontsize=8, ha="center")

        for spine in ax_g.spines.values():
            spine.set_edgecolor("#333340")
        ax_g.tick_params(colors=SUBTEXT)
        plt.tight_layout()
        st.pyplot(fig_gauge)

    with right:
        # Feature importance bars for this prediction
        feature_labels = ["Tempo","Energy","Dance","Valence","Acoustic","Loudness","Speech","Instrum","Duration","Year","Genre","Mode"]
        user_vals = [tempo/200, energy, danceability, valence, acousticness,
                     (loudness_db+30)/30, speechiness, instrumentalness,
                     duration_sec/420, (year-1960)/64, 0.5, 0.5]

        fig_feat, ax_f = plt.subplots(figsize=(6, 4))
        fig_feat.patch.set_facecolor(BG)
        ax_f.set_facecolor(BG)

        feat_importances = gbm_model.feature_importances_
        colors_feat = [GOLD if v > 0.5 else "#A1C9F4" for v in feat_importances]
        bars_f = ax_f.barh(feature_labels, feat_importances * 100, color=colors_feat,
                           edgecolor="none", height=0.65)
        ax_f.set_xlabel("Feature Importance (%)", color=SUBTEXT, fontsize=10)
        ax_f.set_title("Model Feature Importances", color=TEXT, fontsize=12, fontweight="bold", pad=10)
        ax_f.tick_params(colors=TEXT, labelsize=9)
        for spine in ax_f.spines.values():
            spine.set_edgecolor("#333340")
        ax_f.grid(axis="x", color="#333337", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_feat)

    # Score zones legend
    st.markdown(f"""
    <div style="display:flex; gap:24px; margin-top:8px; flex-wrap:wrap;">
      <div style="color:{SUBTEXT};">📍 <b style="color:{SUBTEXT};">0–34</b> Niche Appeal</div>
      <div>📻 <b style="color:#A1C9F4;">35–54</b> Radio Potential</div>
      <div>📈 <b style="color:{GREEN};">55–74</b> Solid Track</div>
      <div>🔥 <b style="color:{GOLD};">75–100</b> Chart Hit</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MOOD ARCHETYPE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🧬 Mood Archetype Classifier")
    st.markdown(f"<p style='color:{SUBTEXT};'>We use K-Means clustering (k=6) on 8 audio features to assign your song to one of 6 emotional archetypes.</p>", unsafe_allow_html=True)

    mood_desc = MOOD_DESCRIPTIONS.get(mood_name, "A unique blend of audio characteristics.")
    mood_emoji = mood_name.split()[0]
    mood_text  = " ".join(mood_name.split()[1:])

    st.markdown(f"""
    <div class="mood-card">
      <div class="mood-emoji">{mood_emoji}</div>
      <div class="mood-name">{mood_text}</div>
      <div class="mood-desc">{mood_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # Radar chart of user's audio profile
    radar_features = ["Energy", "Dance", "Valence", "Acoustic", "Speech", "Instrum"]
    radar_vals = [energy, danceability, valence, acousticness, speechiness, instrumentalness]
    N_r = len(radar_features)
    angles = np.linspace(0, 2 * np.pi, N_r, endpoint=False).tolist()
    angles += angles[:1]
    radar_vals_closed = radar_vals + [radar_vals[0]]

    fig_radar, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig_radar.patch.set_facecolor(BG)
    ax_r.set_facecolor(BG)
    ax_r.spines["polar"].set_color(SUBTEXT)

    ax_r.plot(angles, radar_vals_closed, color=GOLD, linewidth=2.5)
    ax_r.fill(angles, radar_vals_closed, color=GOLD, alpha=0.25)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(radar_features, size=10, color=TEXT)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=7, color=SUBTEXT)
    ax_r.set_ylim(0, 1)
    ax_r.grid(color="#444448", linewidth=0.6)
    ax_r.set_title("Your Song's Audio Fingerprint", size=13, color=TEXT, pad=20, fontweight="bold")
    plt.tight_layout()

    col_radar, col_clusters = st.columns([1, 1])
    with col_radar:
        st.pyplot(fig_radar)

    with col_clusters:
        st.markdown(f"<h4 style='color:{TEXT};'>All 6 Archetypes</h4>", unsafe_allow_html=True)
        for idx, (arch, desc) in enumerate(MOOD_DESCRIPTIONS.items()):
            is_current = (arch == mood_name)
            border_color = GOLD if is_current else "#333340"
            bg_color     = "#2a2a20" if is_current else "#1e1e24"
            star = " ⭐" if is_current else ""
            emoji_part = arch.split()[0]
            name_part  = " ".join(arch.split()[1:])
            st.markdown(f"""
            <div style="background:{bg_color}; border:1px solid {border_color}; border-radius:10px;
                        padding:10px 14px; margin-bottom:8px;">
              <span style="font-size:1.1rem;">{emoji_part}</span>
              <b style="color:{GOLD if is_current else TEXT}; margin-left:6px;">{name_part}{star}</b>
              <div style="color:{SUBTEXT}; font-size:0.8rem; margin-top:4px;">{desc[:80]}...</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DECADE TIME MACHINE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ⏳ Decade Time Machine")
    st.markdown(f"<p style='color:{SUBTEXT};'>How would your song have charted in each decade? We score your audio profile against the sonic norms of each era.</p>", unsafe_allow_html=True)

    decades_list  = list(decade_scores.keys())
    scores_list   = list(decade_scores.values())
    decade_labels = [f"{d}s" for d in decades_list]

    fig_time, ax_t = plt.subplots(figsize=(10, 5))
    fig_time.patch.set_facecolor(BG)
    ax_t.set_facecolor(BG)

    bar_colors_t = [GOLD if d == best_decade else "#A1C9F4" for d in decades_list]
    bars_t = ax_t.bar(decade_labels, scores_list, color=bar_colors_t,
                      width=0.6, edgecolor="none", zorder=3)

    for bar_t, score in zip(bars_t, scores_list):
        ax_t.text(bar_t.get_x() + bar_t.get_width() / 2,
                  bar_t.get_height() + 1, f"{score}", ha="center",
                  va="bottom", color=TEXT, fontsize=11, fontweight="bold")

    ax_t.set_ylim(0, 105)
    ax_t.set_xlabel("Decade", color=SUBTEXT, fontsize=11)
    ax_t.set_ylabel("Projected Chart Score", color=SUBTEXT, fontsize=11)
    ax_t.set_title("Decade Time Machine — Projected Chart Performance", color=TEXT,
                   fontsize=14, fontweight="bold", pad=14)
    ax_t.tick_params(colors=TEXT)
    for spine in ax_t.spines.values():
        spine.set_edgecolor("#333340")
    ax_t.grid(axis="y", color="#333337", linewidth=0.5, alpha=0.5, zorder=0)

    # Best decade annotation
    best_idx = decades_list.index(best_decade)
    ax_t.annotate(f"Best Era 🏆", xy=(best_idx, decade_scores[best_decade]),
                  xytext=(best_idx, decade_scores[best_decade] + 10),
                  ha="center", color=GOLD, fontsize=10, fontweight="bold",
                  arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.5))

    legend_patches_t = [
        mpatches.Patch(color=GOLD, label=f"Best fit: {best_decade}s ({decade_scores[best_decade]})"),
        mpatches.Patch(color="#A1C9F4", label="Other decades"),
    ]
    ax_t.legend(handles=legend_patches_t, facecolor="#2a2a30", edgecolor="#333340",
                labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_time)

    # Decade context cards
    decade_contexts = {
        1960: ("🎸 Motown & Classic Rock Era", "Acoustic warmth, soulful vocals, and raw guitar energy defined the 60s."),
        1970: ("🕺 Disco & Progressive Era",   "Funky basslines, orchestral arrangements, and danceable grooves."),
        1980: ("🎹 Synth-Pop Revolution",       "Big drums, synthesizers, and polished production ruled the airwaves."),
        1990: ("🎤 Grunge & Hip-Hop Golden Age","Raw, authentic emotion met with rhythmic innovation."),
        2000: ("💿 Pop & R&B Dominance",        "Polished production, Auto-Tune experiments, and boy band fever."),
        2010: ("📱 Streaming Era Begins",       "EDM explosion, trap beats, and the rise of viral hits."),
        2020: ("🌐 Genre-Fluid Future",          "Bedroom pop, hyperpop, lo-fi, and genre mashups define the sound."),
    }
    st.markdown("<br>", unsafe_allow_html=True)
    cols_d = st.columns(len(DECADES))
    for col_d, d in zip(cols_d, DECADES):
        ctx_title, ctx_desc = decade_contexts[d]
        score_d = decade_scores[d]
        is_best = (d == best_decade)
        with col_d:
            st.markdown(f"""
            <div style="background:{'#2a2a20' if is_best else '#1e1e24'}; border:1px solid {'#ffd400' if is_best else '#333340'};
                        border-radius:10px; padding:10px 8px; text-align:center;">
              <div style="font-size:1.3rem;">{ctx_title.split()[0]}</div>
              <div style="font-weight:700; color:{GOLD if is_best else TEXT}; font-size:0.9rem;">{d}s</div>
              <div style="font-size:1.4rem; font-weight:800; color:{GOLD if is_best else '#A1C9F4'};">{score_d}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — GENRE RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🎼 Genre Recommender")
    st.markdown(f"<p style='color:{SUBTEXT};'>We measure how closely your audio profile matches the centroid of each genre's distribution. The best-fit genre is your song's natural home.</p>", unsafe_allow_html=True)

    genres_ranked   = [g for g, _ in genre_ranks]
    sim_scores      = [s for _, s in genre_ranks]

    fig_genre, ax_gr = plt.subplots(figsize=(10, 6))
    fig_genre.patch.set_facecolor(BG)
    ax_gr.set_facecolor(BG)

    bar_colors_gr = [GOLD if i == 0 else GENRE_COLORS[i % len(GENRE_COLORS)] for i in range(len(genres_ranked))]
    bars_gr = ax_gr.barh(genres_ranked[::-1], sim_scores[::-1],
                         color=bar_colors_gr[::-1], edgecolor="none", height=0.65)

    for bar_gr, sim in zip(bars_gr, sim_scores[::-1]):
        ax_gr.text(sim + 0.4, bar_gr.get_y() + bar_gr.get_height() / 2,
                   f"{sim}%", va="center", color=TEXT, fontsize=9, fontweight="bold")

    ax_gr.set_xlabel("Genre Similarity (%)", color=SUBTEXT, fontsize=11)
    ax_gr.set_title("Genre Affinity Ranking — Audio Profile Match", color=TEXT,
                    fontsize=14, fontweight="bold", pad=14)
    ax_gr.tick_params(colors=TEXT, labelsize=10)
    ax_gr.set_xlim(0, 115)
    for spine in ax_gr.spines.values():
        spine.set_edgecolor("#333340")
    ax_gr.grid(axis="x", color="#333337", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_genre)

    # Top 3 genre cards
    st.markdown("<br><h4>Top 3 Best-Fit Genres</h4>", unsafe_allow_html=True)
    top3_cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (medal, col_g) in enumerate(zip(medals, top3_cols)):
        g_name, g_sim = genre_ranks[i]
        with col_g:
            st.markdown(f"""
            <div style="background:#23232a; border:1px solid {GENRE_COLORS[i]}; border-radius:12px;
                        padding:18px; text-align:center;">
              <div style="font-size:2rem;">{medal}</div>
              <div style="font-size:1.1rem; font-weight:700; color:{GENRE_COLORS[i]}; margin:6px 0;">{g_name}</div>
              <div style="font-size:1.6rem; font-weight:800; color:{TEXT};">{g_sim}%</div>
              <div style="color:{SUBTEXT}; font-size:0.8rem;">similarity</div>
            </div>
            """, unsafe_allow_html=True)

    # Audio profile comparison vs selected genre
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### 📊 Your Profile vs. Top Genre: **{genres_ranked[0]}**")
    best_g_params = GENRE_PARAMS[genres_ranked[0]]
    compare_feats = ["Tempo (norm)", "Energy", "Danceability", "Valence", "Acousticness", "Speechiness", "Instrumentalness"]
    user_compare  = [tempo/200, energy, danceability, valence, acousticness, speechiness, instrumentalness]
    genre_compare = [
        best_g_params["tempo"][0] / 200,
        best_g_params["energy"][0],
        best_g_params["dance"][0],
        best_g_params["valence"][0],
        best_g_params["acoustic"][0],
        best_g_params["speech"][0],
        best_g_params["instrum"][0],
    ]

    x_comp = np.arange(len(compare_feats))
    width_c = 0.35

    fig_comp, ax_c = plt.subplots(figsize=(10, 4.5))
    fig_comp.patch.set_facecolor(BG)
    ax_c.set_facecolor(BG)

    ax_c.bar(x_comp - width_c/2, user_compare, width_c, color=GOLD, label="Your Song", alpha=0.9, edgecolor="none")
    ax_c.bar(x_comp + width_c/2, genre_compare, width_c, color="#A1C9F4", label=genres_ranked[0], alpha=0.9, edgecolor="none")

    ax_c.set_xticks(x_comp)
    ax_c.set_xticklabels(compare_feats, color=TEXT, fontsize=9, rotation=15, ha="right")
    ax_c.set_ylabel("Score (normalized)", color=SUBTEXT, fontsize=10)
    ax_c.set_title(f"Your Song vs. {genres_ranked[0]} Genre Archetype", color=TEXT, fontsize=12, fontweight="bold", pad=12)
    ax_c.tick_params(colors=TEXT)
    ax_c.legend(facecolor="#2a2a30", edgecolor="#333340", labelcolor=TEXT, fontsize=9)
    for spine in ax_c.spines.values():
        spine.set_edgecolor("#333340")
    ax_c.grid(axis="y", color="#333337", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_comp)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; color:{SUBTEXT}; font-size:0.85rem; padding:12px 0;">
  🎵 <b>Music Intelligence Lab</b> &nbsp;|&nbsp; Powered by Gradient Boosting + K-Means &nbsp;|&nbsp; Built on Zerve
</div>
""", unsafe_allow_html=True)
