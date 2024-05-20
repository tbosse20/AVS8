import numpy as np
import matplotlib.pyplot as plt


def plot_boxplot(numpy_array, file_name):
    
    # Generate random data to test
    # numpy_array = np.random.normal(loc=0, scale=1, size=1000)

    # Plot boxplot
    # plt.boxplot(numpy_array)
    
    # Print the number of samples
    print(f'Cos sim samples: {len(numpy_array)}')
    
    # Plot the distribution
    plt.hist(numpy_array, bins=30, color='skyblue', edgecolor='black')
    
    plt.xlabel("Cosine Similarity Loss Value")
    plt.ylabel("Frequency")
    # plt.title('Cosine Similarity Boxplot')

    # Format the x-axis and y-axis ticks to be integers and reduce the number of ticks
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.savefig(f"output/{file_name}.png")


if __name__ == "__main__":
    
    # Add the number of samples to the file name
    samples = f'_{123}'
    
    # Define the file names
    file_name = f"cosSimilarityLoss_baseline_24.4k_steps_baseline{samples}"
    
    # Load the cosine similarity values
    cos_sims_np = np.load(f"output/{file_name}.npy")

    # Call the function to plot the boxplot
    plot_boxplot(cos_sims_np, file_name)
