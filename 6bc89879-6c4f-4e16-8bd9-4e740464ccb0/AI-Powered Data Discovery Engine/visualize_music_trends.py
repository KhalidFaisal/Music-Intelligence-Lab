
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Zerve palette
BG      = '#1D1D20'
TEXT    = '#fbfbff'
SUBTEXT = '#909094'
COLORS  = ['#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF',
           '#F7B6D2','#1F77B4','#9467BD','#8C564B','#C49C94',
           '#E377C2','#ffd400']

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': SUBTEXT, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': '#333338',
    'font.family': 'sans-serif',
})

df = songs_df.copy()

# ── 1. Loudness War: mean loudness_db per decade ─────────────────────────────
loudness_trend = df.groupby('decade')['loudness_db'].mean().reset_index()

fig1, ax = plt.subplots(figsize=(10, 5))
fig1.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.plot(loudness_trend['decade'], loudness_trend['loudness_db'],
        color='#ffd400', linewidth=2.5, marker='o', markersize=7)
ax.fill_between(loudness_trend['decade'], loudness_trend['loudness_db'],
                loudness_trend['loudness_db'].min() - 0.5,
                color='#ffd400', alpha=0.15)
ax.set_title('The Loudness War: Average Track Loudness by Decade', fontsize=14, color=TEXT, pad=12)
ax.set_xlabel('Decade', fontsize=11)
ax.set_ylabel('Mean Loudness (dBFS)', fontsize=11)
ax.set_xticks(loudness_trend['decade'])
ax.set_xticklabels([f"{d}s" for d in loudness_trend['decade']], rotation=30)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('loudness_war.png', dpi=140, bbox_inches='tight', facecolor=BG)
plt.show()

# ── 2. Acousticness decline over decades ──────────────────────────────────────
ac_trend = df.groupby('decade')['acousticness'].mean().reset_index()

fig2, ax = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.plot(ac_trend['decade'], ac_trend['acousticness'],
        color='#8DE5A1', linewidth=2.5, marker='s', markersize=7)
ax.fill_between(ac_trend['decade'], ac_trend['acousticness'], 0,
                color='#8DE5A1', alpha=0.15)
ax.set_title('Decline of Acousticness Across Decades', fontsize=14, color=TEXT, pad=12)
ax.set_xlabel('Decade', fontsize=11)
ax.set_ylabel('Mean Acousticness (0–1)', fontsize=11)
ax.set_xticks(ac_trend['decade'])
ax.set_xticklabels([f"{d}s" for d in ac_trend['decade']], rotation=30)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('acousticness_decline.png', dpi=140, bbox_inches='tight', facecolor=BG)
plt.show()

# ── 3. Tempo distribution by top genres ──────────────────────────────────────
top_genres = df['genre'].value_counts().head(6).index.tolist()
fig3, ax = plt.subplots(figsize=(10, 5))
fig3.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for i, g in enumerate(top_genres):
    vals = df.loc[df['genre'] == g, 'tempo']
    ax.hist(vals, bins=40, alpha=0.55, label=g, color=COLORS[i], edgecolor='none')
ax.axvline(120, color='#ffd400', linestyle='--', linewidth=1.5, label='120 BPM')
ax.set_title('Tempo Distribution by Genre', fontsize=14, color=TEXT, pad=12)
ax.set_xlabel('Tempo (BPM)', fontsize=11)
ax.set_ylabel('Song Count', fontsize=11)
ax.legend(framealpha=0.2, facecolor=BG, edgecolor=SUBTEXT, labelcolor=TEXT, fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('tempo_distribution.png', dpi=140, bbox_inches='tight', facecolor=BG)
plt.show()

# ── 4. Genre share evolution by decade ───────────────────────────────────────
genre_decade = (df.groupby(['decade', 'genre'])['genre']
                .count()
                .rename('count')
                .reset_index())
genre_decade['share'] = genre_decade.groupby('decade')['count'].transform(lambda x: x / x.sum())
pivot = genre_decade.pivot(index='decade', columns='genre', values='share').fillna(0)
top8 = df['genre'].value_counts().head(8).index.tolist()
pivot = pivot[top8]

fig4, ax = plt.subplots(figsize=(12, 6))
fig4.patch.set_facecolor(BG)
ax.set_facecolor(BG)
pivot.plot(kind='bar', stacked=True, ax=ax,
           color=COLORS[:len(top8)], edgecolor='none', width=0.7)
ax.set_title('Genre Share by Decade', fontsize=14, color=TEXT, pad=12)
ax.set_xlabel('Decade', fontsize=11)
ax.set_ylabel('Share of Songs', fontsize=11)
ax.set_xticklabels([f"{d}s" for d in pivot.index], rotation=30, ha='right')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.legend(title='Genre', framealpha=0.2, facecolor=BG,
          edgecolor=SUBTEXT, labelcolor=TEXT, fontsize=8, title_fontsize=9,
          bbox_to_anchor=(1.01, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('genre_share.png', dpi=140, bbox_inches='tight', facecolor=BG)
plt.show()

# ── 5. Energy & Danceability by decade ───────────────────────────────────────
ed = df.groupby('decade')[['energy', 'danceability']].mean().reset_index()
fig5, ax = plt.subplots(figsize=(10, 5))
fig5.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.plot(ed['decade'], ed['energy'],       color='#FF9F9B', linewidth=2.5, marker='o', label='Energy')
ax.plot(ed['decade'], ed['danceability'], color='#A1C9F4', linewidth=2.5, marker='s', label='Danceability')
ax.set_title('Energy & Danceability Trends by Decade', fontsize=14, color=TEXT, pad=12)
ax.set_xlabel('Decade', fontsize=11)
ax.set_ylabel('Mean Score (0–1)', fontsize=11)
ax.set_xticks(ed['decade'])
ax.set_xticklabels([f"{d}s" for d in ed['decade']], rotation=30)
ax.legend(framealpha=0.2, facecolor=BG, edgecolor=SUBTEXT, labelcolor=TEXT)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('energy_dance_trends.png', dpi=140, bbox_inches='tight', facecolor=BG)
plt.show()

print("All 5 visualizations rendered successfully.")
