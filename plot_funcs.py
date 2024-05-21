import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

folder_path = os.path.join("output", "collected_losses")
    
def save_collected_losses(collected_losses, checkpoint_run, infoNCE_alpha):
    
    os.makedirs(folder_path, exist_ok=True)
    
    for key, value in collected_losses.items():
        file_name = ''
        file_name += f'{key}'
        file_name += f'_{os.path.basename(checkpoint_run)}'
        file_name += f'_baseline' if infoNCE_alpha == 0.0 else f'_clmodel'
        file_path = os.path.join(folder_path, f'{file_name}.npy')
        np.save(file_path, value)
        
def plot_boxplot(xlabel, baseline_array, clmodel_array=None):
    
    # Generate random data to test
    # numpy_array = np.random.normal(loc=0, scale=1, size=1000)
    # numpy_array2 = np.random.normal(loc=0.2, scale=1.2, size=1000)
    
    # Set the style of the visualization
    sns.set(style="whitegrid")
    # Create the density plots
    plt.figure(figsize=(10, 6))
    
    # Plot Baseline historgam and KDE
    print(f'Baseline samples: {len(baseline_array)}')
    sns.kdeplot(baseline_array, color='red', label='Baseline', linewidth=2, fill=True)
    plt.axvline(np.mean(baseline_array), color='r', linestyle='--')
    
    # Plot CLmodel histogram and KDE
    if clmodel_array is not None:
        print(f'CLmodel samples: {len(clmodel_array)}')
        sns.kdeplot(clmodel_array, color='blue', label='CLmodel', linewidth=2, fill=True)
        plt.axvline(np.mean(clmodel_array), color='b', linestyle='--')

    # Configure the plot
    plt.xlabel(xlabel)
    plt.ylabel("Denisty")
    # plt.title('Cosine Similarity Boxplot')
    plt.legend()
    
    # Save the plot
    save_path = os.path.join(folder_path, f"{xlabel.replace(' ', '_')}.png")
    plt.savefig(save_path)


if __name__ == "__main__":
    
    # Define a list of tuples containing the parameters for plotting
    plot_params = [
        {
            'title': 'Cosine Similarity Loss',
            'baseline_file_name': 'custom_sim_loss_baseline_24.4k_steps_baseline.npy',
            'clmodel_file_name': 'custom_sim_loss__clmodel.npy'
        },
        {
            'title': 'Decoder Loss',
            'baseline_file_name': 'custom_decoder_loss_baseline_24.4k_steps_baseline.npy',
            'clmodel_file_name': 'custom_decoder_loss__clmodel.npy'
        }
    ]

    # Iterate over plot_params list
    for params in plot_params:
        # Load the baseline and clmodel losses
        baseline_path = os.path.join(folder_path, params['baseline_file_name'])
        baseline_losses = np.load(baseline_path)
        clmodel_path = os.path.join(folder_path, params['clmodel_file_name'])
        clmodel_losses = np.load(clmodel_path)
        
        # Call the function to plot the boxplot
        plot_boxplot(params['title'], baseline_losses, clmodel_losses)