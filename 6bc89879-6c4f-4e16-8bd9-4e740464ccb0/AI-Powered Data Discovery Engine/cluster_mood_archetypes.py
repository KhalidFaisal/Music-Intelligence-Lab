import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── Zerve Design System ──────────────────────────────────────────────────────
BG       = '#1D1D20'
TEXT     = '#fbfbff'
SUBTEXT  = '#909094'
PALETTE  = ['#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF','#ffd400']

# ── Audio features used for clustering ───────────────────────────────────────
FEATURES = ['tempo','energy','danceability','valence',
            'acousticness','loudness_db','speechiness','instrumentalness']

X_raw = songs_df[FEATURES].copy()

# ── 1. Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── 2. Elbow Method (k = 1 … 12) ─────────────────────────────────────────────
inertias = []
K_range = range(1, 13)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Compute second-derivative to find elbow automatically
diffs = np.diff(inertias, n=2)
elbow_k = int(np.argmin(diffs) + 2)  # offset for 2nd diff
print(f"Elbow auto-detected at k={elbow_k}  (ticket specifies k=6 → using k=6)")

# Elbow plot
fig_elbow, ax_e = plt.subplots(figsize=(8, 4.5))
fig_elbow.patch.set_facecolor(BG)
ax_e.set_facecolor(BG)
ax_e.plot(list(K_range), inertias, color='#A1C9F4', linewidth=2.5, marker='o',
          markersize=6, markerfacecolor='#ffd400', markeredgecolor=BG)
ax_e.axvline(x=6, color='#ffd400', linestyle='--', linewidth=1.5, label='k = 6 (selected)')
ax_e.set_title('Elbow Method — KMeans Inertia vs. k', color=TEXT, fontsize=14, pad=14)
ax_e.set_xlabel('Number of Clusters (k)', color=SUBTEXT, fontsize=11)
ax_e.set_ylabel('Inertia (WCSS)', color=SUBTEXT, fontsize=11)
ax_e.tick_params(colors=SUBTEXT)
for spine in ax_e.spines.values():
    spine.set_edgecolor('#333340')
ax_e.legend(facecolor='#2a2a30', edgecolor='#333340', labelcolor=TEXT, fontsize=10)
plt.tight_layout()
plt.show()

# ── 3. Fit K-Means (k=6) ─────────────────────────────────────────────────────
K = 6
kmeans = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(X_scaled)
songs_df_clustered = songs_df.copy()
songs_df_clustered['cluster_id'] = cluster_labels
print(f"\nCluster sizes:\n{pd.Series(cluster_labels).value_counts().sort_index().to_string()}")

# ── 4. PCA 2-D Reduction ──────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_ * 100
print(f"\nPCA variance explained: PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%  Total={sum(var_exp):.1f}%")

# ── 5. Centroid Profiles ──────────────────────────────────────────────────────
centroids_scaled = kmeans.cluster_centers_
centroids_orig   = scaler.inverse_transform(centroids_scaled)
centroid_df      = pd.DataFrame(centroids_orig, columns=FEATURES)
centroid_df.index.name = 'cluster_id'

# ── 6. Programmatic Mood Archetype Naming ─────────────────────────────────────
def name_cluster(row):
    """Assign a descriptive mood archetype from centroid feature values."""
    energy   = row['energy']
    dance    = row['danceability']
    valence  = row['valence']
    acoustic = row['acousticness']
    speech   = row['speechiness']
    instrum  = row['instrumentalness']
    tempo    = row['tempo']
    loud     = row['loudness_db']

    # Rule-based naming tuned to centroid profiles
    if energy > 0.78 and dance > 0.65 and valence > 0.55:
        return '🔥 Euphoric Dance'
    elif energy > 0.70 and (speech > 0.12 or dance > 0.60) and valence > 0.40:
        return '🎤 High-Energy Hip-Hop'
    elif energy > 0.72 and loud > -7 and acoustic < 0.15 and dance < 0.55:
        return '⚡ Aggressive Rock'
    elif acoustic > 0.35 and energy < 0.60 and valence < 0.52:
        return '🌧 Melancholic Indie'
    elif acoustic > 0.25 and energy < 0.65 and valence > 0.48 and tempo < 120:
        return '🎸 Chill Acoustic'
    elif instrum > 0.25 or (energy > 0.60 and dance > 0.55 and acoustic < 0.20):
        return '🎹 Ambient / Instrumental'
    else:
        return '🎵 Mixed Pop'

cluster_names = {i: name_cluster(centroid_df.loc[i]) for i in range(K)}
print("\n── Mood Archetype Assignments ──────────────────────────────────────────")
for cid, name in cluster_names.items():
    print(f"  Cluster {cid}: {name}")

songs_df_clustered['mood_archetype'] = songs_df_clustered['cluster_id'].map(cluster_names)

# ── 7. Streaming Performance Profiles ────────────────────────────────────────
perf = songs_df_clustered.groupby('cluster_id').agg(
    mood_archetype=('mood_archetype', 'first'),
    n_tracks       =('streams',      'count'),
    avg_streams    =('streams',      'mean'),
    avg_popularity =('popularity',   'mean'),
    avg_energy     =('energy',       'mean'),
    avg_dance      =('danceability', 'mean'),
    avg_valence    =('valence',      'mean'),
    avg_acoustic   =('acousticness', 'mean'),
    avg_tempo      =('tempo',        'mean'),
    avg_loud       =('loudness_db',  'mean'),
).reset_index()

print("\n── Streaming Performance Profiles ──────────────────────────────────────")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
pd.set_option('display.float_format', '{:,.2f}'.format)
print(perf[['cluster_id','mood_archetype','n_tracks','avg_streams',
            'avg_popularity','avg_energy','avg_dance','avg_valence',
            'avg_acoustic','avg_tempo','avg_loud']].to_string(index=False))

# ── 8. 2-D PCA Scatter Plot ───────────────────────────────────────────────────
fig_pca, ax_p = plt.subplots(figsize=(11, 7))
fig_pca.patch.set_facecolor(BG)
ax_p.set_facecolor(BG)

for cid in range(K):
    mask = cluster_labels == cid
    ax_p.scatter(X_pca[mask, 0], X_pca[mask, 1],
                 c=PALETTE[cid], s=8, alpha=0.45, linewidths=0, rasterized=True)

# Plot centroid markers
centroids_pca = pca.transform(centroids_scaled)
for cid in range(K):
    ax_p.scatter(centroids_pca[cid, 0], centroids_pca[cid, 1],
                 c=PALETTE[cid], s=220, marker='*', edgecolors=TEXT,
                 linewidths=0.8, zorder=5)
    ax_p.annotate(
        cluster_names[cid],
        (centroids_pca[cid, 0], centroids_pca[cid, 1]),
        textcoords='offset points', xytext=(10, 6),
        fontsize=9.5, color=TEXT, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a30', edgecolor=PALETTE[cid], alpha=0.85)
    )

legend_patches = [mpatches.Patch(color=PALETTE[i], label=cluster_names[i]) for i in range(K)]
ax_p.legend(handles=legend_patches, loc='lower right', fontsize=9,
            facecolor='#2a2a30', edgecolor='#333340', labelcolor=TEXT)

ax_p.set_title('Mood Archetypes — PCA 2D Projection (k=6 K-Means)',
               color=TEXT, fontsize=14, pad=14)
ax_p.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)', color=SUBTEXT, fontsize=11)
ax_p.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)', color=SUBTEXT, fontsize=11)
ax_p.tick_params(colors=SUBTEXT)
for spine in ax_p.spines.values():
    spine.set_edgecolor('#333340')
plt.tight_layout()
plt.show()

# ── 9. Centroid Radar / Bar Profiles per Cluster ─────────────────────────────
FEAT_SHORT = ['Tempo','Energy','Dance','Valence','Acoustic','Loudness','Speech','Instrum']
# Normalise centroids 0-1 for display
c_min = centroid_df.min()
c_max = centroid_df.max()
centroid_norm = (centroid_df - c_min) / (c_max - c_min + 1e-9)

fig_bars, axes = plt.subplots(2, 3, figsize=(14, 8))
fig_bars.patch.set_facecolor(BG)
fig_bars.suptitle('Centroid Feature Profiles per Mood Archetype',
                  color=TEXT, fontsize=14, y=1.01)

for idx, ax_b in enumerate(axes.flat):
    ax_b.set_facecolor(BG)
    vals = centroid_norm.loc[idx].values
    bars = ax_b.barh(FEAT_SHORT, vals, color=PALETTE[idx], alpha=0.85)
    ax_b.set_xlim(0, 1.05)
    ax_b.set_title(cluster_names[idx], color=PALETTE[idx], fontsize=10, fontweight='bold')
    ax_b.tick_params(colors=SUBTEXT, labelsize=8)
    ax_b.set_xlabel('Normalised value', color=SUBTEXT, fontsize=8)
    for spine in ax_b.spines.values():
        spine.set_edgecolor('#333340')

plt.tight_layout()
plt.show()

print("\n✅ K-Means clustering complete — 6 mood archetypes identified.")
