import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to the uploaded file
file_path = 'analysis/slurm-16_6_clmodel.out'

# Define the regex pattern
pattern = re.compile(
    r"\s+\| > decoder_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > postnet_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > similarity_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > infonce_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > stopnet_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > decoder_coarse_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > decoder_ddc_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > ga_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > decoder_diff_spec_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > postnet_diff_spec_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > decoder_ssim_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > postnet_ssim_loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > loss: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > align_error: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > grad_norm: tensor\((\d+\.\d+),\s*device='cuda:\d+'\)\s+\(tensor\(\d+\.\d+,\s*device='cuda:\d+'\)\)"
    r"\s+\| > current_lr: (\d+\.\d+e[-+]?\d+)\s*\n"
    r"\s+\| > step_time: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
    r"\s+\| > loader_time: (\d+\.\d+) \s+\(\d+\.\d+\)\n"
)

# Initialize an empty list to store the extracted data
data = []

# Read the file and extract loss values
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
    matches = pattern.findall(content)
    for match in matches:
        data.append([float(value) for value in match])

assert len(data) != 0, "No loss values were found in the file."

# Define column names
columns = [
    "decoder_loss", "postnet_loss", "similarity_loss", "infonce_loss", "stopnet_loss",
    "decoder_coarse_loss", "decoder_ddc_loss", "ga_loss", "decoder_diff_spec_loss",
    "postnet_diff_spec_loss", "decoder_ssim_loss", "postnet_ssim_loss", "loss",
    "align_error", "grad_norm", "current_lr", "step_time", "loader_time"
]

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Function to apply a moving average for smoothing
def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# Apply smoothing to each loss column
window_size = 25  # You can adjust the window size for more or less smoothing
smoothed_df = df.apply(lambda x: moving_average(x, window_size) if x.name in columns[:-5] else x)

# Plot each smoothed loss
plt.figure(figsize=(15, 10))

for column in columns[:-5]:  # Exclude the last 5 non-loss columns
    plt.plot(smoothed_df[column], label=column)

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Smoothed Losses Over Steps")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
