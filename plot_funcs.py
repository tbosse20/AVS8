import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot(file_name, baseline_array, slimelime_array=None):
    
    # Set the style of the visualization
    alpha = 0.4 # Transparency of the histograms
    bins = 20 # Number of bins for the histograms
    
    # Generate random data to test
    # numpy_array = np.random.normal(loc=0, scale=1, size=1000)
    # numpy_array2 = np.random.normal(loc=0.2, scale=1.2, size=1000)
    
    # Print the number of samples
    print(f'Baseline samples: {len(baseline_array)}')
    print(baseline_array)
    
    # Set the style of the visualization
    sns.set(style="whitegrid")
    # Create the density plots
    plt.figure(figsize=(10, 6))
    
    # Plot Baseline historgam and KDE
    # sns.histplot(numpy_array, bins=bins, kde=True, color='blue', edgecolor='black', alpha=alpha)
    sns.kdeplot(baseline_array, color='red', label='Baseline', linewidth=2, fill=True)
    plt.axvline(np.mean(baseline_array), color='r', linestyle='--')
    
    # Plot Slimelime histogram and KDE
    if slimelime_array is not None:
        print(f'Slimelime samples: {len(slimelime_array)}')
        # sns.histplot(numpy_array2, bins=bins, kde=True, color="red", edgecolor='black', alpha=alpha)
        sns.kdeplot(slimelime_array, color='blue', label='Slimelime', linewidth=2, fill=True)
        plt.axvline(np.mean(slimelime_array), color='b', linestyle='--')

    # Configure the plot
    plt.xlabel("Cosine Similarity Loss")
    plt.ylabel("Denisty")
    # plt.title('Cosine Similarity Boxplot')
    plt.legend()
    
    # Save the plot
    plt.savefig(f"output/{file_name}.png")


if __name__ == "__main__":
    
    # Define the file names
    file_name = f"cosSimilarityLoss_baseline_24.4k_steps_baseline"
    # Add the number of samples to the file name
    # file_name += f'_{123}'
    
    # Load the cosine similarity values
    cos_sims_np = np.load(f"output/{file_name}.npy")
    cos_sims_np_fake = np.load(f"output/{file_name}.npy") * 1.2

    # Call the function to plot the boxplot
    plot_boxplot(file_name, cos_sims_np, cos_sims_np_fake)
