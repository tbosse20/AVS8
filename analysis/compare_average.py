import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

metric = [
    # 'loader_time',
    'decoder_loss',
    'postnet_loss', 
    'similarity_loss',
    'infonce_loss',
    'stopnet_loss',
    'decoder_coarse_loss',
    'decoder_ddc_loss', 
    'ga_loss',
    'decoder_diff_spec_loss', 
    'postnet_diff_spec_loss',
    'decoder_ssim_loss', 
    'postnet_ssim_loss', 
    'loss',
    'align_error'
]

# Data for CLmodel
clmodel_data = {
    'Metric': metric,
    'CLmodel': [
        # 9.916157188633624,  # avg_loader_time
        1.496860270688539,  # avg_decoder_loss
        1.049160287558661,  # avg_postnet_loss
        1.0932102915898683,  # avg_similarity_loss
        3.0881042891877093,  # avg_infonce_loss
        0.20904734399239389,  # avg_stopnet_loss
        1.466936400427391,  # avg_decoder_coarse_loss
        0.001256453122032648,  # avg_decoder_ddc_loss
        0.0030530489470011967,  # avg_ga_loss
        0.18074299745153247,  # avg_decoder_diff_spec_loss
        0.15799877772460108,  # avg_postnet_diff_spec_loss
        0.4555105560782546,  # avg_decoder_ssim_loss
        0.4023916798173266,  # avg_postnet_ssim_loss
        2.5723555900946478,  # avg_loss
        0.9899604290863209  # avg_align_error
    ]
}

# Data for Baseline
baseline_data = {
    'Metric': metric,
    'Baseline': [
        # 5.60097294488221,  # avg_loader_time
        1.7137772330623153,  # avg_decoder_loss
        0.9246795343510078,  # avg_postnet_loss
        1.0910505636318317,  # avg_similarity_loss
        3.390650556885527,  # avg_infonce_loss
        0.19564336649841188,  # avg_stopnet_loss
        1.6648003024023932,  # avg_decoder_coarse_loss
        0.0013432884363751866,  # avg_decoder_ddc_loss
        0.0030079431318253925,  # avg_ga_loss
        0.17751935424898865,  # avg_decoder_diff_spec_loss
        0.15527568359446892,  # avg_postnet_diff_spec_loss
        0.4474781937509961,  # avg_decoder_ssim_loss
        0.3828787943677445,  # avg_postnet_ssim_loss
        2.698046461212412,  # avg_loss
        0.9902259606564042  # avg_align_error
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