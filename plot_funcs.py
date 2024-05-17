import numpy as np
import matplotlib.pyplot as plt


def plot_boxplot(numpy_array):
    
    # Generate random data to test
    # numpy_array = np.random.normal(loc=0, scale=1, size=1000)

    # Plot boxplot
    # plt.boxplot(numpy_array)
    
    # Plot the distribution
    plt.hist(numpy_array, bins=30, color='skyblue', edgecolor='black')
    
    plt.xlabel("Sample")
    plt.ylabel("Value")
    # plt.title('Cosine Similarity Boxplot')
    plt.savefig("output/cos_similarity_boxplot.png")


if __name__ == "__main__":
    # Load the cosine similarity values
    cos_sims_np = np.load("output/cos_similarity.npy")

    # Call the function to plot the boxplot
    plot_boxplot(cos_sims_np)
