import altair as alt
import pandas as pd
import re

df_corn_pct = pd.read_csv("../data/processed-data/corn_pct_usage.csv")
df_merged = pd.read_csv("../data/processed-data/food_prices.csv")

df_long = df_corn_pct.melt(
    id_vars=['year_quarter'],
    value_vars=['feed_share', 'food_seed_indust_share', 'Fuel_alcohol_share_of_total_use', 'exports_share'],
    var_name='category',
    value_name='percentage'
)

category_labels = {
    'feed_share': 'Feed & Residual',
    'food_seed_indust_share': 'Food, Seed & Industrial',
    'Fuel_alcohol_share_of_total_use': 'Ethanol',
    'exports_share': 'Exports'
}
df_long['category'] = df_long['category'].map(category_labels)

df_long_filtered = df_long[
    (df_long['year_quarter'] >= '2000_Q1') & 
    (df_long['year_quarter'] <= '2015_Q4')
]
df_price_filtered = df_merged[
    (df_merged['year_quarter'] >= '2000_Q1') & 
    (df_merged['year_quarter'] <= '2015_Q4')
]

selection = alt.selection_point(
    fields=['year_quarter'],
    on='mouseover',
    nearest=True,
    empty=False,
    value='2005_Q1'
)

category_colors = alt.Scale(
    domain=['Ethanol', 'Feed & Residual', 'Food, Seed & Industrial', 'Exports'],
    range=['#C9922A', '#2C4A2E', '#6B8E6B', '#8B7355']
)

x_ticks = ['2000_Q1','2002_Q1','2004_Q1','2006_Q1',
            '2008_Q1','2010_Q1','2012_Q1','2014_Q1']

# --- LEFT CHART ---
# Use "container" width so Vega-Embed fills its parent div
base_left = alt.Chart(df_price_filtered).encode(
    x=alt.X('year_quarter:O',
            title=None,
            axis=alt.Axis(labelAngle=-45, values=x_ticks)),
)

price_line = base_left.mark_line(
    color='#C9922A', strokeWidth=2
).encode(
    y=alt.Y('corn_price:Q', title='Corn Price ($/bushel)'),
    tooltip=['year_quarter:O', 'corn_price:Q']
)

rule = base_left.mark_rule(
    color='#2C4A2E', strokeWidth=1, strokeDash=[4, 4]
).encode(
    x=alt.X('year_quarter:O')
).transform_filter(selection)

selected_point = base_left.mark_point(
    color='#2C4A2E', size=120, filled=True
).encode(
    y='corn_price:Q',
    opacity=alt.condition(selection, alt.value(1.0), alt.value(0))
).transform_filter(selection)

overlay = alt.Chart(df_price_filtered).mark_rect(opacity=0).encode(
    x=alt.X('year_quarter:O')
).add_params(selection)

left_chart = (price_line + rule + selected_point + overlay).properties(
    width='container',   # <-- KEY CHANGE: fills parent div
    height=320,
    title=alt.TitleParams('Corn Price Over Time', fontSize=13, font='serif')
)

# --- RIGHT CHART ---
right_chart = alt.Chart(df_long_filtered).mark_bar(size=80).encode(
    x=alt.X('year_quarter:O', axis=None, title=None),
    y=alt.Y('percentage:Q',
            stack='normalize',
            title='Share of Corn Use (%)',
            axis=alt.Axis(format='%')),
    color=alt.Color('category:N',
                   scale=category_colors,
                   title='Use Category',
                   legend=alt.Legend(orient='right')),
    order=alt.Order('category:N'),
    tooltip=[
        alt.Tooltip('category:N', title='Category'),
        alt.Tooltip('percentage:Q', title='Share (%)', format='.1f')
    ]
).transform_filter(
    selection
).properties(
    width=135,           # keep fixed — it's a single-bar snapshot
    height=320,
    title=alt.TitleParams('Corn Use Breakdown', fontSize=13, font='serif')
)

# --- COMBINE ---
chart = alt.hconcat(
    left_chart,
    right_chart,
    spacing=40
).properties(
    background='#F5F0E8',
    title=alt.TitleParams(
        'The Ethanol Squeeze: Corn Price vs. Use Breakdown',
        fontSize=16,
        font='serif'
    )
).configure_axis(
    labelFont='serif',
    titleFont='serif',
    gridColor='#E8E0D0'
).configure_legend(
    labelFont='serif',
    titleFont='serif'
).configure_view(
    strokeWidth=0,
    fill='#F5F0E8'
).configure_title(
    font='serif'
)

# --- SAVE & PATCH HTML ---
output_path = '../outputs/corn_ethanol.html'
chart.save(output_path)

# Inject responsive wrapper styles + a ResizeObserver so Vega rerenders on resize
responsive_patch = """
<style>
  body { margin: 0; padding: 0; }
  #vis { width: 100%; max-width: 100vw; }
  #vis canvas, #vis svg { max-width: 100% !important; }
</style>
<script>
  document.addEventListener("DOMContentLoaded", function () {
    const container = document.getElementById("vis");
    if (!container) return;

    // Vega-Embed exposes the view on the element after render
    const waitForView = setInterval(() => {
      if (container._vgEmbed && container._vgEmbed.view) {
        clearInterval(waitForView);
        const view = container._vgEmbed.view;

        new ResizeObserver(() => {
          // Rerun to let Vega pick up the new container width
          view.runAsync();
        }).observe(container);
      }
    }, 100);
  });
</script>
"""

with open(output_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Insert patch just before </head>
html = html.replace('</head>', responsive_patch + '</head>', 1)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Responsive chart saved to {output_path}")