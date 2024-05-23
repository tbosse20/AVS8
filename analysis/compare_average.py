import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data for CLmodel
clmodel_data = {
    'Metric': [
        'decoder_loss', 'postnet_loss', 
        'similarity_loss', 'infonce_loss', 'stopnet_loss',
        'decoder_coarse_loss', 'decoder_ddc_loss', 'ga_loss',
        'decoder_diff_spec_loss', 'postnet_diff_spec_loss',
        'decoder_ssim_loss', 'postnet_ssim_loss', 'loss',
        'align_error'
    ],
    'CLmodel': [
        3.407711613946546, 1.7241535632873026, 
        1.5607554778735504, 4.221498418994357, 0.5356772717963629,
        3.155016771720998, 0.0008339224229668403, 0.0034714720227412895,
        0.1851753995844828, 0.18158624359946737, 0.4859883688591622,
        0.458419157908513, 4.39831938664284, 0.9919275098941105
    ]
}

# Data for Baseline
baseline_data = {
    'Metric': [
        'decoder_loss', 'postnet_loss', 
        'similarity_loss', 'infonce_loss', 'stopnet_loss',
        'decoder_coarse_loss', 'decoder_ddc_loss', 'ga_loss',
        'decoder_diff_spec_loss', 'postnet_diff_spec_loss',
        'decoder_ssim_loss', 'postnet_ssim_loss', 'loss',
        'align_error'
    ],
    'Baseline': [
        3.471994733612157, 1.7716488218604898, 
        1.6558103856326636, 4.498039897415094, 0.4623983670916726,
        3.3480118733681663, 0.0012244461503619504, 0.0033702696479949914,
        0.17923033063971816, 0.1796362874041972, 0.4767701459773613,
        0.4499532797232487, 4.487329770026738, 0.9915197661529893
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

# Add lines to show differences
for index, row in df.iterrows():
    x1, x2 = row['CLmodel'], row['Baseline']
    y = index
    plt.text(x1 + 0.02, y - 0.15, f"{x1:.3f}")
    plt.text(x2 + 0.02, y + 0.25, f"{x2:.3f}")

plt.title('Performance Comparison: CLmodel vs Baseline')
plt.xlabel('Values')
plt.ylabel('Metrics')
plt.legend(loc='best')
plt.tight_layout()

plt.savefig('analysis/compare_average.png', bbox_inches='tight', dpi=300)