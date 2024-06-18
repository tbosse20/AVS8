import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D

folder_path = os.path.join("analysis", "collected_losses")


def save_collected_losses(collected_losses, infoNCE_alpha, checkpoint_run=None):

    os.makedirs(folder_path, exist_ok=True)

    for key, value in collected_losses.items():
        file_name = ""
        file_name += f"{key}"
        file_name += (
            "_None"
            if checkpoint_run is None
            else f"_{os.path.basename(checkpoint_run)}"
        )
        file_name += f"_baseline" if infoNCE_alpha == 0.0 else f"_clmodel"
        file_path = os.path.join(folder_path, f"{file_name}.npy")
        np.save(file_path, value)


def plot_boxplot(params):

    # Define the x-axis label
    xlabel = params["title"]

    # Define the colors for the plot
    colors = ['r', 'b']

    # Make custom legend lines for versions with colors
    legend_line = [Line2D([0], [0], color=color) for color in colors]
    # Make custom legend names for versions
    legend_name = list(params['files'].keys())
    
    # Add legend lines for trained and untrained versions if they exist
    if len(params['files']['Baseline']) > 1:
        legend_line += [
            Line2D([0], [0], color="black", linestyle="-", alpha=1.0),
            Line2D([0], [0], color="black", linestyle="--", alpha=0.5),
        ]
        legend_name += ['Trained', 'Untrained']

    # Set the style of the visualization
    sns.set(style="whitegrid")
    # Create the density plots
    plt.figure(figsize=(10, 6))

    # Plot Baseline historgam and KDE
    for version_idx, (key, file_names) in enumerate(params["files"].items()):
        for trained_bool, file_name in enumerate(file_names):
            npy_path = os.path.join(folder_path, file_name)
            values = np.load(npy_path)
            sns.kdeplot(
                values,
                color=colors[version_idx],
                linewidth=[2, 2][trained_bool],
                linestyle=["-", "--"][trained_bool],
                alpha=[1.0, 0.5][trained_bool],
            )
            
            # Print mean and std
            mean = np.mean(values)
            std = np.std(values)
            print(f"{key} {'Trained' if trained_bool == 0 else 'Untrained'} {xlabel}:")
            print(f"Mean: {mean:.3f}")
            print(f"Std: {std:.3f}")
            print()

    # Add legend with custom handles
    plt.legend(legend_line, legend_name, loc="upper right")

    # Configure the plot
    plt.xlabel(xlabel)
    plt.ylabel("Denisty")

    # Save the plot
    save_path = os.path.join(folder_path, f"{xlabel.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)


if __name__ == "__main__":

    # Define a list of tuples containing the parameters for plotting
    plot_params = [
        {
            "title": "Cosine Similarity Loss",
            "files": {
                "Baseline": [
                    "custom_sim_loss_50k_baseline_full.npy", # Trained
                    # "custom_sim_loss_None_baseline.npy", # Untrained
                ],
                "CLmodel": [
                    "custom_sim_loss__clmodel_full.npy", # Trained
                    # "custom_sim_loss_None_clmodel.npy", # Untrained
                ],
            },
        },
        {
            "title": "Decoder Loss",
            "files": {
                "Baseline": [
                    "custom_decoder_loss_50k_baseline_full.npy", # Trained
                    # "custom_decoder_loss_None_baseline.npy", # Untrained
                ],
                "CLmodel": [
                    "custom_decoder_loss__clmodel_full.npy", # Trained
                    # "custom_decoder_loss_None_clmodel.npy", # Untrained
                ],
            },
        },
    ]

    # Iterate over plot_params list
    for params in plot_params:

        # Call the function to plot the boxplot
        plot_boxplot(params)
