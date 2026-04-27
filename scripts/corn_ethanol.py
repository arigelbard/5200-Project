"""
corn_ethanol_d3.py

Reads the two processed CSVs and writes a fully self-contained D3 v7 HTML file.
The output is fully responsive — it listens to ResizeObserver and redraws on
every viewport change, so it works embedded in any iframe or page.

Usage:
    python corn_ethanol_d3.py
Output:
    ../outputs/corn_ethanol.html
"""

import pandas as pd
import json
import os

# ── 1. Load & filter data ────────────────────────────────────────────────────

df_corn_pct = pd.read_csv("../data/processed-data/corn_pct_usage.csv")
df_merged   = pd.read_csv("../data/processed-data/food_prices.csv")

# Filter 2000_Q1 → 2015_Q4
def in_range(df):
    return df[(df['year_quarter'] >= '2000_Q1') & (df['year_quarter'] <= '2015_Q4')].copy()

df_price = in_range(df_merged)[['year_quarter', 'corn_price']]

df_use = in_range(df_corn_pct)[
    ['year_quarter', 'feed_share', 'food_seed_indust_share',
     'Fuel_alcohol_share_of_total_use', 'exports_share']
].rename(columns={
    'feed_share':                    'Feed & Residual',
    'food_seed_indust_share':        'Food, Seed & Industrial',
    'Fuel_alcohol_share_of_total_use': 'Ethanol',
    'exports_share':                 'Exports',
})

# ── 2. Serialise to JSON ─────────────────────────────────────────────────────

price_json = json.dumps(df_price.to_dict(orient='records'))
use_json   = json.dumps(df_use.to_dict(orient='records'))

# ── 3. Build the HTML ────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>The Ethanol Squeeze</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
  /* ── Reset & base ── */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #F5F0E8;
    font-family: 'Georgia', serif;
    color: #2C2C2C;
    padding: clamp(12px, 3vw, 32px);
  }}

  #chart-wrapper {{
    width: 100%;
    max-width: 960px;
    margin: 0 auto;
  }}

  h1 {{
    font-size: clamp(14px, 2.5vw, 20px);
    font-weight: normal;
    letter-spacing: 0.02em;
    margin-bottom: 6px;
    color: #2C2C2C;
  }}

  .subtitle {{
    font-size: clamp(11px, 1.5vw, 13px);
    color: #7a7060;
    margin-bottom: 18px;
  }}

  #charts {{
    display: flex;
    gap: clamp(16px, 3vw, 40px);
    align-items: flex-start;
    width: 100%;
  }}

  /* Left chart takes remaining space; right is fixed-ish narrow */
  #left-chart  {{ flex: 1 1 0; min-width: 0; }}
  #right-chart {{ flex: 0 0 clamp(100px, 18%, 155px); }}

  svg {{ display: block; width: 100%; overflow: visible; }}

  /* ── Tooltip ── */
  #tooltip {{
    position: fixed;
    pointer-events: none;
    background: rgba(44,40,34,0.92);
    color: #F5F0E8;
    font-family: 'Georgia', serif;
    font-size: 12px;
    padding: 8px 11px;
    border-radius: 4px;
    line-height: 1.6;
    opacity: 0;
    transition: opacity 0.12s;
    white-space: nowrap;
    z-index: 10;
  }}

  /* ── Legend ── */
  #legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    margin-top: 14px;
    font-size: clamp(10px, 1.4vw, 12px);
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; }}

  /* axis text */
  .axis text {{ font-family: 'Georgia', serif; fill: #5a5040; }}
  .axis path, .axis line {{ stroke: #C8BEA8; }}
  .grid line {{ stroke: #E8E0D0; stroke-dasharray: 3,3; }}
  .grid path {{ stroke: none; }}
</style>
</head>
<body>

<div id="chart-wrapper">
  <h1>The Ethanol Squeeze: Corn Price vs. Use Breakdown</h1>
  <p class="subtitle">Hover the line chart to inspect quarterly corn use composition &nbsp;·&nbsp; 2000 – 2015</p>

  <div id="charts">
    <div id="left-chart"></div>
    <div id="right-chart"></div>
  </div>

  <div id="legend"></div>
</div>

<div id="tooltip"></div>

<script>
// ── Raw data ──────────────────────────────────────────────────────────────
const priceData = {price_json};
const useData   = {use_json};

const CATEGORIES = ['Ethanol','Feed & Residual','Food, Seed & Industrial','Exports'];
const COLORS     = {{'Ethanol':'#C9922A','Feed & Residual':'#2C4A2E',
                    'Food, Seed & Industrial':'#6B8E6B','Exports':'#8B7355'}};

// Quarter labels shown on x-axis (every other year)
const X_TICKS = ['2000_Q1','2002_Q1','2004_Q1','2006_Q1',
                 '2008_Q1','2010_Q1','2012_Q1','2014_Q1'];

function fmtLabel(q) {{
  const [y, qtr] = q.split('_');
  return qtr === 'Q1' ? y : '';
}}

// ── State ─────────────────────────────────────────────────────────────────
let selectedQ = '2005_Q1';
const tooltip = document.getElementById('tooltip');

// ── Legend ────────────────────────────────────────────────────────────────
const legendEl = document.getElementById('legend');
CATEGORIES.forEach(cat => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<div class="legend-swatch" style="background:${{COLORS[cat]}}"></div>
                    <span>${{cat}}</span>`;
  legendEl.appendChild(item);
}});

// ── Draw helpers ──────────────────────────────────────────────────────────

function drawLeft() {{
  const container = document.getElementById('left-chart');
  container.innerHTML = '';

  const totalW = container.clientWidth || 500;
  const margin = {{ top: 24, right: 16, bottom: 52, left: 54 }};
  const W = totalW - margin.left - margin.right;
  const H = Math.round(W * 0.55);   // aspect ratio ~16:9

  const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${{totalW}} ${{H + margin.top + margin.bottom}}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

  const quarters = priceData.map(d => d.year_quarter);

  const xScale = d3.scalePoint()
      .domain(quarters)
      .range([0, W])
      .padding(0.5);

  const yExtent = d3.extent(priceData, d => d.corn_price);
  const yScale = d3.scaleLinear()
      .domain([Math.max(0, yExtent[0] - 0.5), yExtent[1] + 0.5])
      .range([H, 0])
      .nice();

  // Grid
  g.append('g').attr('class','grid')
   .call(d3.axisLeft(yScale).tickSize(-W).tickFormat(''));

  // Axes
  g.append('g').attr('class','axis')
   .attr('transform', `translate(0,${{H}})`)
   .call(d3.axisBottom(xScale)
     .tickValues(X_TICKS)
     .tickFormat(fmtLabel))
   .selectAll('text')
   .attr('transform','rotate(-40)')
   .style('text-anchor','end');

  g.append('g').attr('class','axis')
   .call(d3.axisLeft(yScale).ticks(6).tickFormat(d => `$${{d.toFixed(1)}}`));

  // Y label
  g.append('text')
   .attr('transform','rotate(-90)')
   .attr('x', -H/2).attr('y', -42)
   .attr('text-anchor','middle')
   .style('font-size','11px').style('fill','#7a7060').style('font-family','Georgia,serif')
   .text('Corn Price ($/bushel)');

  // Line
  const line = d3.line()
      .x(d => xScale(d.year_quarter))
      .y(d => yScale(d.corn_price))
      .curve(d3.curveMonotoneX);

  g.append('path')
   .datum(priceData)
   .attr('fill','none')
   .attr('stroke','#C9922A')
   .attr('stroke-width', Math.max(1.5, totalW / 280))
   .attr('d', line);

  // Selected vertical rule
  const rule = g.append('line')
      .attr('stroke','#2C4A2E').attr('stroke-width',1)
      .attr('stroke-dasharray','4,4')
      .attr('y1',0).attr('y2',H);

  // Selected dot
  const dot = g.append('circle')
      .attr('r', Math.max(4, totalW / 100))
      .attr('fill','#2C4A2E').attr('stroke','#F5F0E8').attr('stroke-width',1.5);

  function updateIndicator(q) {{
    const d = priceData.find(r => r.year_quarter === q);
    if (!d) return;
    const x = xScale(q), y = yScale(d.corn_price);
    rule.attr('x1', x).attr('x2', x);
    dot.attr('cx', x).attr('cy', y);
  }}
  updateIndicator(selectedQ);

  // Invisible overlay for hover
  const bisect = d3.bisector(d => d.year_quarter).center;

  svg.append('rect')
     .attr('fill','transparent')
     .attr('x', margin.left).attr('y', margin.top)
     .attr('width', W).attr('height', H)
     .on('mousemove', function(event) {{
       const [mx] = d3.pointer(event, this);
       // Find nearest quarter by x position
       let best = quarters[0], bestDist = Infinity;
       quarters.forEach(q => {{
         const dist = Math.abs(xScale(q) - mx);
         if (dist < bestDist) {{ bestDist = dist; best = q; }}
       }});
       if (best !== selectedQ) {{
         selectedQ = best;
         updateIndicator(selectedQ);
         drawRight();
       }}
       // Tooltip
       const pd = priceData.find(r => r.year_quarter === selectedQ);
       const ud = useData.find(r => r.year_quarter === selectedQ);
       if (!pd) return;
       const [px, py] = d3.pointer(event);
       const total = CATEGORIES.reduce((s,c) => s + (ud?.[c] || 0), 0);
       let html = `<strong>${{selectedQ.replace('_',' ')}}</strong><br>
                   Price: <strong>$${{pd.corn_price.toFixed(2)}}/bu</strong>`;
       if (ud) {{
         html += '<br>──────────────';
         CATEGORIES.forEach(c => {{
           const pct = total > 0 ? (ud[c]/total*100).toFixed(1) : '–';
           html += `<br><span style="color:${{COLORS[c]}}">■</span> ${{c}}: ${{pct}}%`;
         }});
       }}
       tooltip.innerHTML = html;
       tooltip.style.opacity = '1';
       tooltip.style.left = (event.clientX + 14) + 'px';
       tooltip.style.top  = (event.clientY - 10) + 'px';
     }})
     .on('mouseleave', () => {{ tooltip.style.opacity = '0'; }});
}}

function drawRight() {{
  const container = document.getElementById('right-chart');
  container.innerHTML = '';

  const totalW = container.clientWidth || 140;
  const margin = {{ top: 24, right: 10, bottom: 8, left: 46 }};
  const W = totalW - margin.left - margin.right;
  // Match height of left chart by recomputing the same aspect ratio
  const leftW = document.getElementById('left-chart').clientWidth || 500;
  const leftH = Math.round((leftW - 16 - 54) * 0.55);
  const H = leftH;

  const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${{totalW}} ${{H + margin.top + margin.bottom}}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

  const ud = useData.find(r => r.year_quarter === selectedQ);
  if (!ud) return;

  const total = CATEGORIES.reduce((s,c) => s + (ud[c] || 0), 0);
  let cumY = 0;
  const segments = CATEGORIES.map(c => {{
    const val = ud[c] || 0;
    const frac = total > 0 ? val / total : 0;
    const seg = {{ category: c, frac, y0: cumY, y1: cumY + frac }};
    cumY += frac;
    return seg;
  }});

  const yScale = d3.scaleLinear().domain([0,1]).range([H,0]);

  g.append('g').attr('class','axis')
   .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.0%')))
   .selectAll('text').style('font-size','10px');

  // Y-axis title
  g.append('text')
   .attr('transform','rotate(-90)')
   .attr('x', -H/2).attr('y', -38)
   .attr('text-anchor','middle')
   .style('font-size','10px').style('fill','#7a7060').style('font-family','Georgia,serif')
   .text('Share of Corn Use');

  segments.forEach(seg => {{
    g.append('rect')
     .attr('x', 0).attr('width', W)
     .attr('y', yScale(seg.y1))
     .attr('height', Math.max(0, yScale(seg.y0) - yScale(seg.y1)))
     .attr('fill', COLORS[seg.category]);
  }});

  // Chart title
  svg.append('text')
     .attr('x', margin.left + W/2).attr('y', 14)
     .attr('text-anchor','middle')
     .style('font-size','11px').style('fill','#2C2C2C').style('font-family','Georgia,serif')
     .text('Use Breakdown');
}}

// ── Initial draw ──────────────────────────────────────────────────────────
drawLeft();
drawRight();

// ── Responsive redraw ─────────────────────────────────────────────────────
let resizeTimer;
new ResizeObserver(() => {{
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {{ drawLeft(); drawRight(); }}, 60);
}}).observe(document.getElementById('chart-wrapper'));
</script>
</body>
</html>
"""

# ── 4. Write output ──────────────────────────────────────────────────────────

out_path = '../outputs/corn_ethanol.html'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Saved → {out_path}")