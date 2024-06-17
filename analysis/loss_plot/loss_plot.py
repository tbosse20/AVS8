import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import seaborn as sns
import numpy as np

folder_path = os.path.join("analysis", "loss_plot")

# Concatenate the data for each run
def concat_run_names(df: pd.DataFrame, run_names: list, key_name: str):
    collected = [
        df[f"{run_name} - {key_name}"]
        for run_name in run_names
        if f"{run_name} - {key_name}" in df.columns
    ]

    if len(collected) == 0:
        return np.array([])

    return pd.concat(collected, axis=0, ignore_index=True).dropna().values


def plot_loss(
    version_run_names: dict,
    key_names: list,
    smooth: int = 25,
    raw: bool = False,
    show_ratio: bool = False,
    verbose: bool = False,
):
    """
    Plot the specific loss from a CSV file as collective in terms of "baseline" and "CLmodel".

    Args:
        version_run_names (dict): Dictionary of version names and run names.
        key_names (list): List of key names to plot.
        smooth  (int): Window size smoothing.
        raw (bool): Show the raw plot (un-smooth).
        verbose (bool): Show the plot.

    Example:
        plot_loss(version_run_names, ["loss", "decoder_loss"], smooth=25)

    Setup:
        1. Update the "version_run_names" with the run names.
        2. Download all csv files from wandb export
        3. Rename them "wandb {loss_name}.csv"
    """

    # Define colors
    colors = ["blue", "red"]
    # Define line style
    linestyles = [
        "-",
        (0, (5, 10)),
        (0, (1, 10)),
    ]

    # Set the style of the visualization
    sns.set(style="whitegrid")
    # Plot the data
    plt.figure(figsize=(10, 6))

    # Make custom legend lines for versions with colors
    legend_line = [Line2D([0], [0], color=color) for color in colors]
    # Make custom legend names for versions
    legend_name = list(version_run_names.keys())

    # List to compare two versions
    ratio_list = []
    
    # Iterate over given key names
    for key_name_idx, key_name in enumerate(key_names):

        # Read CSV file into a DataFrame
        csv_path = os.path.join(folder_path, f"wandb {key_name}.csv")
        df = pd.read_csv(csv_path, sep=",", quotechar='"')

        # Plot lines for each series
        for version_idx, (version, run_names) in enumerate(version_run_names.items()):

            # Concatenate the data for each run
            values = concat_run_names(df, run_names, key_name)
            
            # Limit plot to 24k steps
            values = values[:24000]

            # Plot values to ratio
            if show_ratio:
                ratio_list.append(values)

            # Calculate and plot moving average smoothing
            smooth_values = pd.DataFrame(values).rolling(smooth).mean()
            plt.plot(
                smooth_values,
                color=colors[version_idx],
                linestyle=linestyles[key_name_idx],
                linewidth=1,
            )

            # Skip if raw line not selected
            if raw:
                plt.plot(
                    values,
                    color=colors[version_idx],
                    alpha=0.2,
                )

        # Skip if only one key name
        if len(key_names) > 1:
            # Add key names to legend with different line style
            legend_line.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=linestyles[key_name_idx],
                    linewidth=0.9,
                )
            )
            # Add key names to legend with capitalized
            legend_name.append(key_name.capitalize().replace("_", " "))

    # Plot loss ratio
    if show_ratio:
        # Shorten to the shortest length
        shortest = min(len(ratio_list[0]), len(ratio_list[1]))
        ratio_list[0] = ratio_list[0][:shortest]
        ratio_list[1] = ratio_list[1][:shortest]
        ratio = ratio_list[0] / ratio_list[1]
        smooth_ratio = pd.DataFrame(ratio).rolling(smooth).mean()

        # Plot loss ratio
        plt.plot(smooth_ratio, color="green")
        # Add to legend with custom handles
        legend_line.append(Line2D([0], [0], color="green"))
        # Add to legend with custom name
        legend_name.append("Loss Ratio")
        
        # Annotate the plot with the values
        for i in range(0, len(smooth_ratio), 2500):
            value = smooth_ratio.iloc[i]
            plt.annotate(
                f"{value[0]:.2f}",  # Format the number to 2 decimal places
                (i, value),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=10,
                color="black"
            )
                
    # Add legend with custom handles
    plt.legend(legend_line, legend_name, loc="upper right")
        
    # Configure plot
    plt.xlabel("Step")
    plt.ylabel("Loss Value")
    # plt.title(f"{track} over Steps")
    plt.grid(True)

    # Show plot
    if verbose:
        plt.show()

    # Save plot
    file_path = os.path.join(folder_path, f"{key_names}.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=300)


def process_all_csv(version_run_names: dict, smooth: int = 25):
    """
    Plot in respect to "plot_together_list", CSV files in the "/analysis".

    Args:
        smooth (int): Window size smoothing.

    Example:
        process_all_csv(smooth=100)
    """

    plot_together_list = [
        ["decoder_ssim_loss", "postnet_ssim_loss"],
        ["ga_loss", "decoder_ddc_loss"],
        ["decoder_loss"],
        ["stopnet_loss", "decoder_diff_spec_loss"],
        ["postnet_diff_spec_loss", "postnet_loss"],
        ["similarity_loss", "infonce_loss"],
    ]

    for plot_together in plot_together_list:
        print(f"Processing: {plot_together}")
        plot_loss(version_run_names, key_names=plot_together, smooth=smooth)


def rename_csv_files(folder_name):

    # Loop through all files in the folder
    for file_name in os.listdir(folder_name):
        # Skip if not a CSV file
        if not (file_name.endswith(".csv") and file_name.startswith("wandb_")):
            continue

        # Get the file path
        file_path = os.path.join(folder_name, file_name)

        # Read file
        df = pd.read_csv(file_path, sep=",", quotechar='"')
        # Get the second column name
        column_name = df.columns[1].split(" ")[-1]
        # Get the new file name
        new_file_name = f'{file_name.split("_")[0]} {column_name}.csv'
        # Rename the file
        print(f"Renaming: {file_name} -> {new_file_name}")
        os.rename(file_path, os.path.join(folder_name, new_file_name))


# Example usage
if __name__ == "__main__":

    # Define baseline and CLmodel run names
    version_run_names = {
        "Baseline": [
            # Testing baseline runs
            # "graceful-night-317",
            # "proud-plant-319",
            # "fast-field-321",
            # Actual baseline runs
            "rare-snowball-314",
            "ethereal-feather-315",
            "graceful-night-317",
            "proud-plant-319",
            "fast-field-321",
            "breezy-haze-337",
            "quiet-flower-343",
            "quiet-haze-348",
            "fanciful-lion-357",
            "restful-grass-361",
            "atomic-moon-368",
            "toasty-sound-371",  # 24k steps
            # "lunar-capybara-372",
            # "clear-wildflower-374",
            # "expert-forest-388",
            # "wandering-moon-412",
            # "sage-meadow-418",
            # "toasty-galaxy-420",
            # "summer-snowball-434",
        ],
        "CLmodel": [
            # Testing baseline runs
            # "breezy-elevator-264",
            # "copper-sun-265",
            # "lunar-grass-318",
            # "fresh-totem-319",
            # "peachy-lake-322",
            # "good-breeze-347",
            # Actual CLmodel runs
            "revived-durian-275",
            "worthy-water-353",
            "cosmic-sky-356",
            "spring-sun-358",
            "polar-wave-360",
            "fresh-thunder-368",
            "astral-thunder-370",
            "light-water-372",
            "youthful-sea-374",
            "denim-capybara-388",
            "wandering-vortex-412",
            "efficient-cherry-417",
            "celestial-moon-419",
            "autumn-salad-420",
            "glorious-forest-429",  # 24k steps
        ],
    }

    # rename_csv_files("analysis")

    # Process all files
    process_all_csv(version_run_names, smooth=2500)

    # Process singe file
    plot_loss(version_run_names, ["loss"], smooth=100, raw=True, show_ratio=True)
