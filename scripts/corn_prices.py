import matplotlib.pyplot as plt
import pandas as pd

df_merged = pd.read_csv("../data/processed-data/food_prices.csv")

# Change gridlines, background color, and font family of visual to match website
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#F5F0E8')
ax.set_facecolor('#F5F0E8')
ax.grid(color='#E8E0D0', linewidth=0.5)

# Set labels
ax.set_title('Food Commodity Prices Indexed to 2005')
ax.set_ylabel('Price Index (2005 = 100)')

# Fix x-axis to show only every 4th quarter (every year)
quarters = df_merged['year_quarter'].tolist()
tick_positions = range(0, len(quarters), 4)
tick_labels = [quarters[i] for i in tick_positions]

# Rotate x-axis labels
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45)

# Each foodstuff color and line thickness
ax.plot(df_merged['year_quarter'], df_merged['poultry_idx'], 
        label='Poultry', color="#5873CB", linewidth=1)

ax.plot(df_merged['year_quarter'], df_merged['soybean_idx'], 
        label='Soybeans', color="#2C4A2E", linewidth=1)

ax.plot(df_merged['year_quarter'], df_merged['pork_idx'], 
        label='Pork', color="#8B4513", linewidth=1)

ax.plot(df_merged['year_quarter'], df_merged['corn_idx'], 
        label='Corn', color="#C9922A", linewidth=2)

# Energy policy act annotation line
ax.axvline(x='2005_Q1', color='Black', linestyle='--', linewidth=1)
ax.text('2004_Q3', 175, 'Energy Policy Act (2005)', 
        color='black', fontsize=10, rotation=90)

# EISA annotation line
ax.axvline(x='2007_Q1', color='Black', linestyle='--', linewidth=1)
ax.text('2006_Q3', 195, 'EISA (2007)', 
        color='black', fontsize=10, rotation=90)

# EPA reduction mandate annotation line
ax.axvline(x='2013_Q3', color='Black', linestyle='--', linewidth=1)
ax.text('2013_Q1', 105, 'EPA reduces mandate (2013)', 
        color='black', fontsize=10, rotation=90)

# Horizontal line for line with 100 index
ax.axhline(y=100, color="#919085", linestyle='--', linewidth=.5)
ax.text('90', 40,'', color='Black', fontsize=10, rotation=0)

# Shaded area for 2012 crop drought
ax.axvspan('2012_Q2', '2012_Q4', alpha=0.15, color='#8B4513', label='2012 Drought')

ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()