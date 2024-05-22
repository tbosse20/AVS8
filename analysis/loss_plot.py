import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import seaborn as sns
import numpy as np

# Concatenate the data for each run
def concat_run_names(df: pd.DataFrame, run_names: list, key_name: str):
    collected = [
                df[f"{run_name} - {key_name}"]
                for run_name in run_names
                if f"{run_name} - {key_name}" in df.columns
            ]
    return pd.concat(collected, axis=0, ignore_index=True).dropna().values


def plot_loss(
    version_run_names: dict,
    key_names: list,
    smooth: int = 25,
    raw: bool = False,
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
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
    ]

    # Set the style of the visualization
    sns.set(style="whitegrid")
    # Plot the data
    plt.figure(figsize=(10, 6))

    # Make custom legend lines for versions with colors
    legend_line = [Line2D([0], [0], color=color) for color in colors]
    # Make custom legend names for versions
    legend_name = list(version_run_names.keys())

    # Get dir files
    valid_key_names = []

    # Iterate over given key names
    for key_name_idx, key_name in enumerate(key_names):

        # Read CSV file into a DataFrame
        try:
            csv_path = os.path.join("analysis", f"wandb {key_name}.csv")
            df = pd.read_csv(csv_path, sep=",", quotechar='"')
            valid_key_names.append(key_name)
        except FileNotFoundError:
            continue

        # Plot lines for each series
        for version_idx, (version, run_names) in enumerate(version_run_names.items()):

            # Concatenate the data for each run
            values = concat_run_names(df, run_names, key_name)

            # Calculate and plot moving average smoothing
            smooth_values = pd.DataFrame(values).rolling(smooth).mean()
            plt.plot(
                smooth_values,
                color=colors[version_idx],
                linestyle=linestyles[key_name_idx],
            )

            # Skip if raw line not selected
            if raw:
                plt.plot(
                    values,
                    color=colors[version_idx],
                    alpha=0.2,
                )

        # Add key names to legend with different line style
        legend_line.append(
            Line2D([0], [0], color="black", linestyle=linestyles[key_name_idx])
        )
        # Add key names to legend with capitalized
        legend_name.append(key_name.replace("_", " "))

    # Skip if no valid key names
    if len(valid_key_names) == 0:
        return

    # Update key names with valid key names
    key_names = valid_key_names

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
    plt.savefig(f"analysis/{key_names}.png")


def process_all_csv(version_run_names: dict, smooth: int = 25):
    """
    Plot in respect to "plot_together_list", CSV files in the "/analysis".

    Args:
        smooth (int): Window size smoothing.

    Example:
        process_all_csv(smooth=100)
    """

    plot_together_list = [
        [
            "ga_loss",
            "post_ssim_loss",
            "postnet_diff_spec_loss",
            "stopnet_loss",
            "postnet_loss",
        ],
        [
            "decoder_loss",
            "decoder_ddc_loss",
            "decoder_ssim_loss",
            "decoder_diff_spec_loss",
        ],
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
            "toasty-sound-371", # 24k steps
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
            "glorious-forest-429", # 24k steps
        ],
    }
    
    rename_csv_files("analysis")

    # Process all files
    # process_all_csv(version_run_names, smooth=100)

    # Process singe file
    # plot_loss(version_run_names, ["loss"], smooth=50, raw=True)
