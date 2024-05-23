import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data for CLmodel
clmodel_data = {
    'Metric': [
        'loader_time', 'decoder_loss', 'postnet_loss', 
        'similarity_loss', 'infonce_loss', 'stopnet_loss',
        'decoder_coarse_loss', 'decoder_ddc_loss', 'ga_loss',
        'decoder_diff_spec_loss', 'postnet_diff_spec_loss',
        'decoder_ssim_loss', 'postnet_ssim_loss', 'loss',
        'align_error'
    ],
    'CLmodel': [
        5.85770615569767, 3.407711613946546, 1.7241535632873026, 
        1.5607554778735504, 4.221498418994357, 0.5356772717963629,
        3.155016771720998, 0.0008339224229668403, 0.0034714720227412895,
        0.1851753995844828, 0.18158624359946737, 0.4859883688591622,
        0.458419157908513, 4.39831938664284, 0.9919275098941105
    ]
}

# Data for Baseline
baseline_data = {
    'Metric': [
        'loader_time', 'decoder_loss', 'postnet_loss', 
        'similarity_loss', 'infonce_loss', 'stopnet_loss',
        'decoder_coarse_loss', 'decoder_ddc_loss', 'ga_loss',
        'decoder_diff_spec_loss', 'postnet_diff_spec_loss',
        'decoder_ssim_loss', 'postnet_ssim_loss', 'loss',
        'align_error'
    ],
    'Baseline': [
        5.604639457317995, 3.4719955980653823, 1.7716483703026404, 
        1.6558248278009173, 0.0, 0.4623984035236176,
        3.3480122094342715, 0.0012244470635324478, 0.0033702695647374625,
        0.17923032720098872, 0.17963628427526318, 0.4767700650587895,
        0.44995327191640333, 2.9488673958609843, 0.9915197612340184
    ]
}

# Custom colors for bars
palette = {'CLmodel': 'blue', 'Baseline': 'red'}

# Create DataFrame
df_clmodel = pd.DataFrame(clmodel_data)
df_baseline = pd.DataFrame(baseline_data)

# Merge DataFrames on Metric
df = pd.merge(df_clmodel, df_baseline, on='Metric')

# Calculate differences
df['Difference'] = df['CLmodel'] - df['Baseline']

# Plotting
plt.figure(figsize=(14, 10))
sns.set(style="whitegrid")

# Plot CLmodel and Baseline
df_melted = df.melt(id_vars='Metric', value_vars=['CLmodel', 'Baseline'], var_name='Model', value_name='Value')
sns.barplot(x='Value', y='Metric', hue='Model', data=df_melted, palette=palette)

# # Add lines to show differences
# for index, row in df.iterrows():
#     x1, x2 = row['CLmodel'], row['Baseline']
#     y = index
#     color = 'green' if row['Difference'] < 0 else 'red'
#     plt.plot([x1, x2], [y, y], color=color, lw=5, solid_capstyle='butt')
#     plt.text(max(x1, x2) + 0.15, y + 0.1, f"{row['Difference']:.2f}", color='black', ha='center')

plt.title('Performance Comparison: CLmodel vs Baseline')
plt.xlabel('Values')
plt.ylabel('Metrics')
plt.legend(loc='best')
plt.tight_layout()

plt.savefig('analysis/compare_average.png', bbox_inches='tight', dpi=300)