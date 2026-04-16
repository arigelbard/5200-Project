# %% Install packages
import matplotlib.pyplot as plt

# %% Carbon emissions bar chart
fig, ax = plt.subplots(figsize=(8, 6))

categories = ['EPA Projection\n(Corn Ethanol)', 'Gasoline\nBaseline', 'Searchinger et al.\n(Corn Ethanol)']
values     = [74, 92, 177]
colors     = ['#2a6496', '#555555', '#c0392b']

bars = ax.bar(categories, values, color=colors, width=0.5, zorder=3)

# ── Value labels on top of each bar ───────────────────────────────────
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f'{val} gCO₂eq/MJ',
        ha='center', va='bottom',
        fontsize=10, fontweight='bold',
        color=bar.get_facecolor()
    )

# ── Difference annotations ─────────────────────────────────────────────
# Arrow and label showing gap between EPA and Searchinger
ax.annotate(
    '',
    xy=(2, 177), xytext=(2, 74),
    arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.5)
)
ax.text(2.28, 125, '+103\ngCO₂eq/MJ\ndifference',
        fontsize=8.5, color='#c0392b', va='center')

# ── Gasoline reference line ────────────────────────────────────────────
ax.axhline(y=92, color='#555555', linewidth=1,
           linestyle='--', alpha=0.5, zorder=2)

# ── Formatting ─────────────────────────────────────────────────────────
ax.set_ylabel('Lifecycle GHG emissions (gCO₂eq/MJ)', fontsize=11)
ax.set_title(
    'Lifecycle GHG Emissions: What the Mandate Promised vs. Reality\n'
    'Corn ethanol per MJ compared to gasoline baseline',
    fontsize=12, fontweight='bold', pad=12
)
ax.set_ylim(0, 210)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Source note ────────────────────────────────────────────────────────
fig.text(
    0.12, 0.01,
    'Sources: EPA RFS2 Regulatory Impact Analysis (2010); '
    'Searchinger et al., Science (2008)',
    fontsize=7.5, color='#888888'
)

plt.tight_layout()
plt.savefig('../outputs/carbon_emissions_bar.png', dpi=150, bbox_inches='tight')
print("Chart saved to ../outputs/carbon_emissions_bar.png")
plt.show()
# %%
