
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ── Zerve palette ─────────────────────────────────────────────────────────────
_BG      = '#1D1D20'
_TEXT    = '#fbfbff'
_SUBTEXT = '#909094'
_GOLD    = '#ffd400'
_GREEN   = '#17b26a'
_RED     = '#f04438'
_COLORS  = ['#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF',
            '#F7B6D2','#1F77B4','#9467BD','#8C564B','#C49C94',
            '#E377C2','#ffd400']

plt.rcParams.update({
    'figure.facecolor': _BG, 'axes.facecolor': _BG,
    'axes.edgecolor': _SUBTEXT, 'axes.labelcolor': _TEXT,
    'xtick.color': _TEXT, 'ytick.color': _TEXT,
    'text.color': _TEXT, 'grid.color': '#333338',
    'font.family': 'sans-serif',
})

_df = songs_df.copy()
_decades = sorted(_df['decade'].unique())
_decade_labels = [f"{d}s" for d in _decades]

# ════════════════════════════════════════════════════════════════════════════════
# CHART 1: Audio feature evolution — energy, valence, acousticness, tempo (normalized)
# ════════════════════════════════════════════════════════════════════════════════
_feature_means = _df.groupby('decade')[['energy','valence','acousticness','tempo']].mean()
# Normalize tempo to 0-1 scale for overlay
_tempo_norm = (_feature_means['tempo'] - _feature_means['tempo'].min()) / \
              (_feature_means['tempo'].max() - _feature_means['tempo'].min())

audio_evolution_fig = plt.figure(figsize=(12, 6))
audio_evolution_fig.patch.set_facecolor(_BG)
_ax1 = audio_evolution_fig.add_subplot(111)
_ax1.set_facecolor(_BG)

_feat_colors = {'energy': '#FF9F9B', 'valence': '#A1C9F4', 'acousticness': '#8DE5A1'}
for _feat, _col in _feat_colors.items():
    _vals = _feature_means[_feat]
    _ax1.plot(_feature_means.index, _vals, color=_col, linewidth=2.5,
              marker='o', markersize=7, label=_feat.capitalize())
    _ax1.fill_between(_feature_means.index, _vals, alpha=0.08, color=_col)

# Tempo on secondary axis
_ax1_twin = _ax1.twinx()
_ax1_twin.set_facecolor(_BG)
_ax1_twin.plot(_feature_means.index, _feature_means['tempo'], color=_GOLD,
               linewidth=2.0, marker='D', markersize=6, linestyle='--', label='Tempo (BPM)')
_ax1_twin.set_ylabel('Mean Tempo (BPM)', color=_GOLD, fontsize=10)
_ax1_twin.tick_params(axis='y', colors=_GOLD)
_ax1_twin.spines['right'].set_edgecolor(_GOLD)

_ax1.set_title('Audio Feature Evolution by Decade', fontsize=15, color=_TEXT, pad=14, fontweight='bold')
_ax1.set_xlabel('Decade', fontsize=11)
_ax1.set_ylabel('Mean Score (0–1)', fontsize=11)
_ax1.set_xticks(_decades)
_ax1.set_xticklabels(_decade_labels, rotation=0)
_ax1.grid(axis='y', linestyle='--', alpha=0.35)

# Combined legend
_handles1, _labels1 = _ax1.get_legend_handles_labels()
_handles2, _labels2 = _ax1_twin.get_legend_handles_labels()
_ax1.legend(_handles1 + _handles2, _labels1 + _labels2,
            framealpha=0.2, facecolor=_BG, edgecolor=_SUBTEXT, labelcolor=_TEXT, fontsize=9,
            loc='upper right')
plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# CHART 2: The Loudness War — dBFS rise per decade with annotation
# ════════════════════════════════════════════════════════════════════════════════
_loud_by_decade = _df.groupby('decade')['loudness_db'].agg(['mean','std']).reset_index()
_db_delta = _loud_by_decade['mean'].iloc[-1] - _loud_by_decade['mean'].iloc[0]

loudness_war_fig = plt.figure(figsize=(11, 5))
loudness_war_fig.patch.set_facecolor(_BG)
_ax2 = loudness_war_fig.add_subplot(111)
_ax2.set_facecolor(_BG)

_ax2.fill_between(_loud_by_decade['decade'],
                   _loud_by_decade['mean'] - _loud_by_decade['std'],
                   _loud_by_decade['mean'] + _loud_by_decade['std'],
                   alpha=0.15, color=_GOLD)
_ax2.plot(_loud_by_decade['decade'], _loud_by_decade['mean'],
          color=_GOLD, linewidth=3, marker='o', markersize=9, zorder=5)

# Label each decade value
for _, _row in _loud_by_decade.iterrows():
    _ax2.annotate(f"{_row['mean']:.1f} dB",
                  xy=(_row['decade'], _row['mean']),
                  xytext=(0, 14), textcoords='offset points',
                  ha='center', fontsize=9, color=_TEXT)

_ax2.set_title(f'The Loudness War: +{abs(_db_delta):.1f} dB Rise Since 1960s',
               fontsize=14, color=_TEXT, pad=14, fontweight='bold')
_ax2.set_xlabel('Decade', fontsize=11)
_ax2.set_ylabel('Mean Loudness (dBFS)', fontsize=11)
_ax2.set_xticks(_loud_by_decade['decade'])
_ax2.set_xticklabels(_decade_labels, rotation=0)
_ax2.grid(axis='y', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# CHART 3: Happiest & Saddest Decades by Valence
# ════════════════════════════════════════════════════════════════════════════════
_valence_by_decade = _df.groupby('decade')['valence'].agg(['mean','sem']).reset_index()
_happiest_idx = _valence_by_decade['mean'].idxmax()
_saddest_idx  = _valence_by_decade['mean'].idxmin()
_happiest_dec = _valence_by_decade.loc[_happiest_idx, 'decade']
_saddest_dec  = _valence_by_decade.loc[_saddest_idx,  'decade']

_bar_colors = [_RED if _row['decade'] == _saddest_dec
               else _GREEN if _row['decade'] == _happiest_dec
               else '#5a5a60'
               for _, _row in _valence_by_decade.iterrows()]

valence_decades_fig = plt.figure(figsize=(11, 5))
valence_decades_fig.patch.set_facecolor(_BG)
_ax3 = valence_decades_fig.add_subplot(111)
_ax3.set_facecolor(_BG)

_bars = _ax3.bar(_valence_by_decade['decade'], _valence_by_decade['mean'],
                 color=_bar_colors, width=7, zorder=3,
                 yerr=_valence_by_decade['sem'] * 1.96, capsize=4,
                 error_kw={'ecolor': _SUBTEXT, 'linewidth': 1.2})

for _bar, (_, _row) in zip(_bars, _valence_by_decade.iterrows()):
    _ax3.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.005,
              f"{_row['mean']:.3f}", ha='center', va='bottom', fontsize=9, color=_TEXT)

_ax3.set_title('Happiest & Saddest Decades by Average Valence (Emotional Positivity)',
               fontsize=13, color=_TEXT, pad=14, fontweight='bold')
_ax3.set_xlabel('Decade', fontsize=11)
_ax3.set_ylabel('Mean Valence (0–1)', fontsize=11)
_ax3.set_xticks(_valence_by_decade['decade'])
_ax3.set_xticklabels(_decade_labels, rotation=0)
_ax3.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)

_legend_patches = [
    mpatches.Patch(color=_GREEN, label=f'Happiest: {_happiest_dec}s'),
    mpatches.Patch(color=_RED,   label=f'Saddest: {_saddest_dec}s'),
    mpatches.Patch(color='#5a5a60', label='Other decades'),
]
_ax3.legend(handles=_legend_patches, framealpha=0.2, facecolor=_BG,
            edgecolor=_SUBTEXT, labelcolor=_TEXT, fontsize=9)
plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# CHART 4: Correlation heatmap — features vs popularity
# ════════════════════════════════════════════════════════════════════════════════
_feat_cols = ['energy','danceability','valence','acousticness','loudness_db',
              'speechiness','instrumentalness','tempo','duration_sec']
_corr_matrix = _df[_feat_cols + ['popularity']].corr()

# Sort features by absolute correlation with popularity
_pop_corrs = _corr_matrix['popularity'][_feat_cols].sort_values(key=abs, ascending=False)

pop_correlation_fig = plt.figure(figsize=(11, 6))
pop_correlation_fig.patch.set_facecolor(_BG)
_ax4 = pop_correlation_fig.add_subplot(111)
_ax4.set_facecolor(_BG)

_bar_colors_corr = [_GREEN if v > 0 else _RED for v in _pop_corrs.values]
_bars4 = _ax4.barh(_pop_corrs.index[::-1], _pop_corrs.values[::-1],
                   color=_bar_colors_corr[::-1], height=0.6, zorder=3)
_ax4.axvline(0, color=_SUBTEXT, linewidth=1, linestyle='--')

for _bar, _val in zip(_bars4, _pop_corrs.values[::-1]):
    _x_offset = 0.005 if _val >= 0 else -0.005
    _ha = 'left' if _val >= 0 else 'right'
    _ax4.text(_val + _x_offset, _bar.get_y() + _bar.get_height() / 2,
              f"{_val:+.3f}", va='center', ha=_ha, fontsize=9, color=_TEXT)

_ax4.set_title('Feature Correlations with Track Popularity', fontsize=14, color=_TEXT, pad=14, fontweight='bold')
_ax4.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
_ax4.grid(axis='x', linestyle='--', alpha=0.35, zorder=0)
_legend_corr = [
    mpatches.Patch(color=_GREEN, label='Positive correlation'),
    mpatches.Patch(color=_RED,   label='Negative correlation'),
]
_ax4.legend(handles=_legend_corr, framealpha=0.2, facecolor=_BG,
            edgecolor=_SUBTEXT, labelcolor=_TEXT, fontsize=9)
plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# CHART 5: Genre Audio Fingerprints — Spider/Radar Charts
# ════════════════════════════════════════════════════════════════════════════════
_radar_features = ['energy','danceability','valence','acousticness','speechiness','instrumentalness']
_radar_labels = ['Energy','Danceability','Valence','Acousticness','Speechiness','Instrumentalness']
_radar_genres = ['Pop','Rock','Hip-Hop','Electronic/Dance','R&B/Soul','Country',
                 'Jazz','Classical','Folk/Acoustic','Metal','Latin','Indie']

_num_vars = len(_radar_features)
_angles = np.linspace(0, 2 * np.pi, _num_vars, endpoint=False).tolist()
_angles += _angles[:1]  # close the polygon

# Build 3x4 radar grid
radar_fingerprints_fig, _axes_radar = plt.subplots(3, 4, figsize=(18, 13),
                                                    subplot_kw=dict(polar=True))
radar_fingerprints_fig.patch.set_facecolor(_BG)
radar_fingerprints_fig.suptitle('Genre Audio DNA — Feature Fingerprints',
                                 fontsize=17, color=_TEXT, y=1.01, fontweight='bold')

for _idx, (_genre_name, _ax_r) in enumerate(zip(_radar_genres, _axes_radar.flatten())):
    _ax_r.set_facecolor(_BG)
    _ax_r.spines['polar'].set_color(_SUBTEXT)

    _genre_data = _df[_df['genre'] == _genre_name][_radar_features].mean().values.tolist()
    _genre_data += _genre_data[:1]

    _ax_r.plot(_angles, _genre_data, color=_COLORS[_idx], linewidth=2)
    _ax_r.fill(_angles, _genre_data, color=_COLORS[_idx], alpha=0.22)

    # Style
    _ax_r.set_xticks(_angles[:-1])
    _ax_r.set_xticklabels(_radar_labels, size=8, color=_TEXT)
    _ax_r.set_yticks([0.2, 0.4, 0.6, 0.8])
    _ax_r.set_yticklabels(['0.2','0.4','0.6','0.8'], size=6, color=_SUBTEXT)
    _ax_r.tick_params(axis='x', pad=7)
    _ax_r.set_ylim(0, 1)
    _ax_r.grid(color='#444448', linewidth=0.6)
    _ax_r.set_title(_genre_name, size=11, color=_COLORS[_idx], pad=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════════
# PRINTED INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("  🎵  MUSIC DEEP EDA — KEY INSIGHTS")
print("=" * 68)

# Audio evolution
_e1960 = _feature_means.loc[1960, 'energy']
_e2020 = _feature_means.loc[2020, 'energy']
_v1960 = _feature_means.loc[1960, 'valence']
_v2020 = _feature_means.loc[2020, 'valence']
_a1960 = _feature_means.loc[1960, 'acousticness']
_a2020 = _feature_means.loc[2020, 'acousticness']
_t1960 = _feature_means.loc[1960, 'tempo']
_t2020 = _feature_means.loc[2020, 'tempo']
print("\n📈  AUDIO EVOLUTION (1960s → 2020s)")
print(f"   Energy:       {_e1960:.3f} → {_e2020:.3f}  ({'+' if _e2020>_e1960 else ''}{_e2020-_e1960:+.3f})")
print(f"   Valence:      {_v1960:.3f} → {_v2020:.3f}  ({_v2020-_v1960:+.3f})")
print(f"   Acousticness: {_a1960:.3f} → {_a2020:.3f}  ({_a2020-_a1960:+.3f})")
print(f"   Tempo:        {_t1960:.1f} → {_t2020:.1f} BPM  ({_t2020-_t1960:+.1f} BPM)")

# Loudness war
_loud1960 = _loud_by_decade[_loud_by_decade['decade']==1960]['mean'].values[0]
_loud2020 = _loud_by_decade[_loud_by_decade['decade']==2020]['mean'].values[0]
print("\n🔊  THE LOUDNESS WAR")
print(f"   1960s avg loudness:  {_loud1960:.2f} dBFS")
print(f"   2020s avg loudness:  {_loud2020:.2f} dBFS")
print(f"   Total dB rise:       {abs(_loud2020-_loud1960):.2f} dB louder over 60 years")
_t_stat_loud, _p_loud = stats.pearsonr(_df['decade'], _df['loudness_db'])
print(f"   Correlation (decade–loudness): r={_t_stat_loud:.3f}, p<0.001" if _p_loud < 0.001
      else f"   Correlation: r={_t_stat_loud:.3f}, p={_p_loud:.4f}")

# Valence / emotional decades
print("\n😊😢  HAPPIEST & SADDEST DECADES (by Valence)")
_valence_ranked = _valence_by_decade.sort_values('mean', ascending=False)
for _, _row in _valence_ranked.iterrows():
    _marker = " ◀ HAPPIEST" if _row['decade'] == _happiest_dec \
              else " ◀ SADDEST"  if _row['decade'] == _saddest_dec else ""
    print(f"   {int(_row['decade'])}s:  valence = {_row['mean']:.4f}{_marker}")

# Correlation with popularity
print("\n📊  TOP FEATURE CORRELATIONS WITH POPULARITY")
for _feat, _corr_val in _pop_corrs.items():
    _bar_str = '█' * int(abs(_corr_val) * 40)
    _sign = '+' if _corr_val > 0 else '−'
    print(f"   {_feat:20s}: {_sign}{abs(_corr_val):.3f}  {_bar_str}")

# Genre fingerprint extremes
print("\n🎸  GENRE AUDIO DNA — EXTREMES")
_genre_means = _df.groupby('genre')[_radar_features].mean()
for _feat in _radar_features:
    _top = _genre_means[_feat].idxmax()
    _bot = _genre_means[_feat].idxmin()
    print(f"   {_feat:20s}: highest = {_top:20s} ({_genre_means.loc[_top, _feat]:.3f})  |  "
          f"lowest = {_bot:20s} ({_genre_means.loc[_bot, _feat]:.3f})")

print("\n" + "=" * 68)
print("  ✅  Deep EDA complete — 5 publication-quality charts rendered")
print("=" * 68)
