import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_loss(csv_file: str, smooth: int = 25, verbose: bool = False):
    """
    Plot the specific loss from a CSV file as collective in terms of "baseline" and "slimelime".

    Args:
        csv_file (str): Path to the CSV file.
        smooth  (int):  Window size smoothing.
        verbose (bool): Show the plot.
    """

    # Define colors for each run
    colors = {"baseline": "blue", "slimelime": "red"}

    # Define baseline and slimelime run names
    baseline_run_names = [
        # Testing baseline runs
        "graceful-night-317",
        "proud-plant-319",
        "fast-field-321",
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
    ]
    slimelime_run_names = [
        # Testing baseline runs
        "breezy-elevator-264",
        "copper-sun-265",
        "revived-durian-275",
        "lunar-grass-318",
        "fresh-totem-319",
        "peachy-lake-322",
        # Actual slimelime runs
        "good-breeze-347",
        "worthy-water-353",
        "cosmic-sky-356",
        "spring-sun-358",
        "polar-wave-360",
    ]

    # Extract track name from file name
    track = csv_file.split(" ")[1][:-4]

    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file, sep=",", quotechar='"').dropna()

    def concat_run_names(run_names):
        return pd.concat(
            [
                df[f"{run_name} - {track}"]
                for run_name in run_names
                if f"{run_name} - {track}" in df.columns
            ],
            axis=0,
        ).values

    # Concatenate the data for each run
    baseline = concat_run_names(baseline_run_names)
    slimelime = concat_run_names(slimelime_run_names)

    # Plot the data
    plt.figure(figsize=(10, 6))

    # Plot lines for each series
    plt.plot(baseline, label="Baseline", color=colors["baseline"], alpha=0.2)
    plt.plot(slimelime, label="Slimelime", color=colors["slimelime"], alpha=0.2)

    # Calculate and plot moving average smoothing
    smooth_baseline = pd.DataFrame(baseline).rolling(smooth).mean()
    smooth_slimelime = pd.DataFrame(slimelime).rolling(smooth).mean()
    plt.plot(smooth_baseline, color=colors["baseline"])
    plt.plot(smooth_slimelime, color=colors["slimelime"])

    # Set legend with custom handles
    plt.legend(
        [
            Line2D([0], [0], color=colors["baseline"], alpha=1, linewidth=2),
            Line2D([0], [0], color=colors["slimelime"], alpha=1, linewidth=2),
        ],
        ["Baseline", "Slimelime"],
        loc="upper right",
    )

    # Configure plot
    plt.xlabel("Step")
    plt.ylabel(track.capitalize().replace("_", " "))
    # plt.title(f"{track} over Steps")
    plt.grid(True)

    # Show plot
    if verbose:
        plt.show()

    # Save plot
    plt.savefig(f"analysis/{track}.png")


def process_all_csv(smooth: int = 25):
    import os

    for file in os.listdir("analysis"):
        if file.endswith(".csv"):
            print(f"Processing {file}")
            plot_loss(f"analysis/{file}", smooth=smooth)


# Example usage
if __name__ == "__main__":

    # Process all files
    process_all_csv(smooth=100)

    # Process singe file
    # plot_loss("analysis/wandb loss.csv", window_size=25)
