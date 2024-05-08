import numpy as np
import matplotlib.pyplot as plt

def plot_boxplot(numpy_array, save_path):
    
    # Plot boxplot
    plt.boxplot(numpy_array)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Boxplot of Tensors')
    plt.savefig(save_path)

if __name__ == "__main__":
    # Load the cosine similarity values
    cos_sims_np = np.load('output/cos_similarity.npy')
    # Call the function to plot the boxplot
    plot_boxplot(cos_sims_np, 'output/cos_similarity_boxplot.png')