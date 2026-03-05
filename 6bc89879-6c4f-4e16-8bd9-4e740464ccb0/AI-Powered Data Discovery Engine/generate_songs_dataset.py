
import numpy as np
import pandas as pd

np.random.seed(42)
N = 10_000

# ── Genre definitions ──────────────────────────────────────────────────────────
genres = ['Pop', 'Rock', 'Hip-Hop', 'Electronic/Dance', 'R&B/Soul',
          'Country', 'Jazz', 'Classical', 'Folk/Acoustic', 'Metal',
          'Latin', 'Indie']

# Genre weights shift over decades – we'll assign genre per song later
# but first build per-decade genre probability tables
decade_list = [1960, 1970, 1980, 1990, 2000, 2010, 2020]

# Proportion of each genre per decade (rows=decades, cols=genres, order matches `genres`)
genre_decade_probs = np.array([
    # Pop   Rock  HH    Elec  R&B   Cntry Jazz  Class Folk  Metal Latin Indie
    [0.20, 0.25, 0.01, 0.01, 0.12, 0.10, 0.10, 0.06, 0.09, 0.00, 0.04, 0.02],  # 1960
    [0.20, 0.28, 0.03, 0.03, 0.12, 0.08, 0.07, 0.04, 0.08, 0.01, 0.04, 0.02],  # 1970
    [0.25, 0.22, 0.06, 0.08, 0.10, 0.07, 0.04, 0.03, 0.04, 0.03, 0.05, 0.03],  # 1980
    [0.22, 0.17, 0.14, 0.10, 0.10, 0.07, 0.03, 0.02, 0.03, 0.04, 0.05, 0.03],  # 1990
    [0.24, 0.12, 0.18, 0.12, 0.10, 0.08, 0.02, 0.01, 0.03, 0.03, 0.05, 0.02],  # 2000
    [0.22, 0.09, 0.20, 0.16, 0.11, 0.07, 0.01, 0.01, 0.03, 0.02, 0.06, 0.02],  # 2010
    [0.20, 0.07, 0.22, 0.14, 0.12, 0.06, 0.01, 0.01, 0.03, 0.02, 0.09, 0.03],  # 2020
])

# ── Year & decade assignment ───────────────────────────────────────────────────
# More songs from recent decades (streaming era)
decade_weights = np.array([0.05, 0.07, 0.10, 0.13, 0.17, 0.25, 0.23])
decade_idx = np.random.choice(len(decade_list), size=N, p=decade_weights)
decade_year = np.array(decade_list)[decade_idx]
year = decade_year + np.random.randint(0, 10, size=N)
year = np.clip(year, 1960, 2024)
decade = (year // 10) * 10

# ── Genre per song ─────────────────────────────────────────────────────────────
genre = np.array([
    np.random.choice(genres, p=genre_decade_probs[d])
    for d in decade_idx
])

# ── Helper: per-genre base parameters ─────────────────────────────────────────
# (tempo_mean, tempo_std, energy_mean, dance_mean, valence_mean, acoustic_mean,
#  speech_mean, inst_mean, loud_mean)
genre_params = {
    'Pop':              (118, 14, 0.72, 0.70, 0.60, 0.22, 0.05, 0.03, -5.5),
    'Rock':             (128, 18, 0.82, 0.52, 0.48, 0.14, 0.05, 0.04, -6.5),
    'Hip-Hop':          (92,  14, 0.68, 0.76, 0.55, 0.10, 0.22, 0.02, -6.0),
    'Electronic/Dance': (128, 12, 0.85, 0.80, 0.58, 0.05, 0.06, 0.15, -5.0),
    'R&B/Soul':         (96,  16, 0.62, 0.68, 0.52, 0.20, 0.08, 0.03, -6.5),
    'Country':          (108, 16, 0.62, 0.60, 0.65, 0.38, 0.05, 0.01, -7.5),
    'Jazz':             (110, 25, 0.50, 0.55, 0.62, 0.48, 0.05, 0.25, -10.0),
    'Classical':        (100, 30, 0.28, 0.25, 0.50, 0.85, 0.03, 0.80, -15.0),
    'Folk/Acoustic':    (108, 18, 0.42, 0.52, 0.58, 0.72, 0.05, 0.12, -11.0),
    'Metal':            (140, 22, 0.90, 0.42, 0.35, 0.06, 0.06, 0.08, -5.5),
    'Latin':            (108, 16, 0.76, 0.80, 0.72, 0.18, 0.08, 0.04, -6.0),
    'Indie':            (112, 18, 0.60, 0.60, 0.52, 0.38, 0.05, 0.08, -8.0),
}

# ── Decade modifiers (trend adjustments) ──────────────────────────────────────
# Loudness war: loudness increases ~0.4 dB per decade from 1960 baseline
# Acousticness: declining ~0.06 per decade
# Energy: slight increase ~0.015 per decade
# Danceability: slight increase for modern era
decade_norm = (decade - 1960) / 10  # 0..6

loud_decade_bonus   = decade_norm * 0.45        # +0 to +2.7 dB
acoustic_decade_pen = decade_norm * 0.055        # −0 to −0.33
energy_decade_bonus = decade_norm * 0.012        # +0 to +0.072
dance_decade_bonus  = decade_norm * 0.010        # +0 to +0.06

# ── Sample raw features ────────────────────────────────────────────────────────
tempo        = np.zeros(N)
energy       = np.zeros(N)
danceability = np.zeros(N)
valence      = np.zeros(N)
acousticness = np.zeros(N)
loudness     = np.zeros(N)
speechiness  = np.zeros(N)
instrumentalness = np.zeros(N)

for g in genres:
    mask = (genre == g)
    n_g  = mask.sum()
    p = genre_params[g]
    tempo[mask]        = np.random.normal(p[0], p[1], n_g)
    energy[mask]       = np.random.beta(a=max(0.5, p[2]*5), b=max(0.5, (1-p[2])*5), size=n_g)
    danceability[mask] = np.random.beta(a=max(0.5, p[3]*5), b=max(0.5, (1-p[3])*5), size=n_g)
    valence[mask]      = np.random.beta(a=max(0.5, p[4]*5), b=max(0.5, (1-p[4])*5), size=n_g)
    acousticness[mask] = np.random.beta(a=max(0.5, p[5]*6), b=max(0.5, (1-p[5])*6), size=n_g)
    loudness[mask]     = np.random.normal(p[8], 2.5, n_g)
    speechiness[mask]  = np.random.beta(a=max(0.5, p[6]*4), b=max(0.5, (1-p[6])*4), size=n_g)
    instrumentalness[mask] = np.random.beta(a=max(0.3, p[7]*3), b=max(0.3, (1-p[7])*3), size=n_g)

# Apply decade trends
energy       = np.clip(energy       + energy_decade_bonus,  0.0, 1.0)
danceability = np.clip(danceability + dance_decade_bonus,   0.0, 1.0)
acousticness = np.clip(acousticness - acoustic_decade_pen,  0.0, 1.0)
loudness     = loudness + loud_decade_bonus
tempo        = np.clip(tempo, 40, 220)
speechiness  = np.clip(speechiness, 0.0, 0.96)
instrumentalness = np.clip(instrumentalness, 0.0, 1.0)
valence      = np.clip(valence, 0.0, 1.0)

# ── Popularity score (0–100) ───────────────────────────────────────────────────
# Influenced by energy, danceability, recency, genre
recency_bonus = decade_norm * 5  # up to +30 for 2020s
popularity_raw = (
    40
    + energy       * 20
    + danceability * 15
    + valence      * 5
    + recency_bonus
    + np.random.normal(0, 8, N)
)
# Genre-based nudge
genre_pop_nudge = {
    'Pop': 8, 'Hip-Hop': 6, 'Electronic/Dance': 4, 'Latin': 5,
    'R&B/Soul': 3, 'Rock': 2, 'Country': 1, 'Indie': 0,
    'Folk/Acoustic': -2, 'Metal': -1, 'Jazz': -4, 'Classical': -6,
}
for g, nudge in genre_pop_nudge.items():
    popularity_raw[genre == g] += nudge

popularity = np.clip(np.round(popularity_raw).astype(int), 0, 100)

# ── Streaming counts ───────────────────────────────────────────────────────────
# Log-normal; strongly correlated with popularity and recency
log_streams_mean = 11 + (popularity / 100) * 6 + recency_bonus * 0.25
streams_raw = np.random.lognormal(mean=log_streams_mean, sigma=1.4, size=N).astype(int)
# Pre-streaming era songs get far fewer streams
pre_streaming = year < 2000
streams_raw[pre_streaming] = (streams_raw[pre_streaming] * 0.05).astype(int)
streams = np.clip(streams_raw, 0, 5_000_000_000)

# ── Duration (seconds) ────────────────────────────────────────────────────────
# Pop / hip-hop trend toward shorter tracks in 2010s+
duration_base = np.random.normal(220, 40, N)
# Shorter tracks in modern era: −8s per decade from 1990 onward
shortening = np.where(decade >= 1990, (decade - 1990) / 10 * 8, 0)
duration_s = np.clip(duration_base - shortening, 60, 600).astype(int)

# ── Key & mode ────────────────────────────────────────────────────────────────
key = np.random.choice(list(range(12)), size=N)   # 0=C … 11=B
mode = np.random.choice([0, 1], size=N, p=[0.38, 0.62])  # 0=minor, 1=major

key_names  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
mode_names = {0: 'Minor', 1: 'Major'}

# ── Assemble DataFrame ────────────────────────────────────────────────────────
songs_df = pd.DataFrame({
    'year':              year,
    'decade':            decade,
    'genre':             genre,
    'key':               [key_names[k] for k in key],
    'mode':              [mode_names[m] for m in mode],
    'tempo':             np.round(tempo, 1),
    'energy':            np.round(energy, 4),
    'danceability':      np.round(danceability, 4),
    'valence':           np.round(valence, 4),
    'acousticness':      np.round(acousticness, 4),
    'loudness_db':       np.round(loudness, 2),
    'speechiness':       np.round(speechiness, 4),
    'instrumentalness':  np.round(instrumentalness, 4),
    'duration_sec':      duration_s,
    'popularity':        popularity,
    'streams':           streams,
})

# ── QA summary ────────────────────────────────────────────────────────────────
print(f"Dataset shape: {songs_df.shape}")
print(f"\nGenre distribution:\n{songs_df['genre'].value_counts().to_string()}")
print(f"\nDecade distribution:\n{songs_df['decade'].value_counts().sort_index().to_string()}")
print(f"\nFeature ranges:")
for col in ['tempo','energy','danceability','valence','acousticness',
            'loudness_db','speechiness','instrumentalness','popularity']:
    print(f"  {col:20s}: [{songs_df[col].min():.3f}, {songs_df[col].max():.3f}]  "
          f"mean={songs_df[col].mean():.3f}  std={songs_df[col].std():.3f}")
print(f"\nStreams: min={songs_df['streams'].min():,}  "
      f"max={songs_df['streams'].max():,}  "
      f"median={songs_df['streams'].median():,.0f}")
print(f"\nNull counts:\n{songs_df.isnull().sum().to_string()}")
print("\nSample rows:")
print(songs_df.sample(5, random_state=7).to_string())
