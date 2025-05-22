#!/usr/bin/env python3
# coding: utf-8

import os

os.environ["OMP_NUM_THREADS"] = "1"

import random
import glob
import argparse
#from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import euclidean

import datetime
import sys

from PyPDF2 import PdfReader, PdfWriter

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Helper Functions ###

def print_package_versions():
    logger.info(f"numpy: {np.__version__}")
    logger.info(f"pandas: {pd.__version__}")
    logger.info('matplotlib: {}'.format(matplotlib.__version__))
    # logger.info(f"scipy: {scipy.__version__}")
    #logger.info(f"scipy: {euclidean.__module__.split('.')[0]} (v{euclidean.__module__.split('.')[2]})")

def check_for_nan(matrix):
    """Check if the matrix contains any NaN values."""
    return matrix.isnull().values.any() or np.any(np.isnan(matrix.values))

def load_matrix(file_path):
    """Load the matrix from a file."""
    return pd.read_csv(file_path, sep=r'\s+')

def create_plot_directories(base_path, method, k_iter):
    """
    Create subdirectories for each clustering method and k_iter value.

    Parameters:
    - base_path: The base plot path.
    - method: The clustering method (e.g., 'GMM', 'PCA').
    - k_iter: The current k_iter value.

    Returns:
    - method_plot_path: The full path to the created subdirectory.
    """
    method_plot_path = os.path.join(base_path, method, f'k{k_iter}')
    os.makedirs(method_plot_path, exist_ok=True)
    return method_plot_path

def save_probs_ids_tsv(probabilities, sample_names, distances, path, n_components, filename_prefix="cluster_report_k"):
    """
    Sort the probabilities and sample names, assign IDs, calculate distances, and save them to a TSV file.
    
    Parameters:
    - probabilities: 2D numpy array where each row contains the probabilities of each sample belonging to each cluster.
    - sample_names: List of sample names corresponding to the rows in the probabilities matrix.
    - distances: Dictionary mapping sample names to their distance from the cluster center.
    - path: Directory path to save the TSV file.
    - n_components: The number of clusters (components) in GMM.
    - filename_prefix: Prefix for the TSV filename (default: 'cluster_probabilities_ids_k').
    """
    # Sort samples by cluster assignment and max probability in descending order
    assigned_clusters = np.argmax(probabilities, axis=1)  # Determine assigned clusters
    sorted_indices = np.lexsort((-np.max(probabilities, axis=1), assigned_clusters))  # Sort by cluster and max probability
    # Reorder probabilities, sample names, and calculate aligned fragments
    probabilities_sorted = probabilities[sorted_indices]
    sample_names_sorted = np.array(sample_names)[sorted_indices]
    # Extract numbers after 'n' in sample names for aligned fragments
    sample_mapping_nums = np.array([extract_number_after_n(name) for name in sample_names_sorted])
    # Create sample IDs
    sample_ids = np.arange(1, len(sample_names_sorted) + 1)
    sample_ids_str = [f'{id:0{len(str(len(sample_names)))}d}' for id in sample_ids]  # Pad IDs with leading zeros
    # Create filename based on the number of components (clusters)
    filename = f"{filename_prefix}{n_components}.tsv"
    # Create column names for the clusters
    n_clusters = probabilities.shape[1]
    cluster_columns = [f'Cluster{i+1}' for i in range(n_clusters)]
    # Create a pandas DataFrame to store the cluster probabilities
    df = pd.DataFrame(probabilities_sorted, columns=cluster_columns)
    # Insert additional columns
    df.insert(0, 'SampleID', sample_ids_str)
    df.insert(1, 'SampleName', sample_names_sorted)
    df.insert(2, "AlignedFrags", sample_mapping_nums)
    # Add the Distance column
    df['ClusterCenterDist'] = df['SampleName'].map(distances)
    # Save the DataFrame to a TSV file
    tsv_path = f'{path}/{filename}'
    df.to_csv(tsv_path, sep='\t', index=False)

def scale_pdf(input_pdf_path, output_pdf_path, target_width):
    """
    Scale the input PDF pages to a specified width while preserving aspect ratio.
    """
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        scale_factor = target_width / width
        new_width = width * scale_factor
        new_height = height * scale_factor
        page.scale_to(new_width, new_height)
        writer.add_page(page)
    with open(output_pdf_path, "wb") as output_pdf_file:
        writer.write(output_pdf_file)

def merge_pdfs(pdf_list, output_path):
    """
    Merge multiple PDFs into a single PDF file.
    """
    pdf_writer = PdfWriter()
    for pdf in pdf_list:
        pdf_reader = PdfReader(pdf)
        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])
    with open(output_path, "wb") as output_pdf:
        pdf_writer.write(output_pdf)

def perform_pca(matrix, n_components):
    """
    Perform PCA on the input matrix and return the transformed data.
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(matrix)
    explained_variance = pca.explained_variance_ratio_
    return transformed_data, explained_variance, pca

def create_truncated_colormap(base_color, n=100):
    """
    Create a truncated colormap from a base color by transitioning it from light to the full color.
    """
    colors = [(1, 1, 1), base_color]
    new_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n)
    return new_cmap

def plot_pca_gradient(transformed_data, sample_names, probabilities, path='.', explained_variance=None, pdf_filename="pca_gradient_with_distances.pdf"):
    """
    Plot PCA results with color gradient based on cluster assignment, using transparency to indicate probabilities.
    """
    n_clusters = probabilities.shape[1]
    tab10 = plt.get_cmap('tab10')
    colors = tab10.colors[:n_clusters]  # Base colors for each cluster
    cluster_assignments = np.argmax(probabilities, axis=1)
    pc1 = transformed_data[:, 0]
    pc2 = transformed_data[:, 1]
    # Create a truncated colormap for each cluster based on its color
    cmap_dict = {
        cluster: create_truncated_colormap(colors[cluster])
        for cluster in range(n_clusters)
    }
    # Calculate minimum probabilities per cluster
    #min_probabilities = [probabilities[cluster_assignments == cluster, cluster].min() for cluster in range(n_clusters)]
    min_probabilities = [
        probabilities[cluster_assignments == cluster, cluster].min()
        if np.any(cluster_assignments == cluster) else 0
        for cluster in range(n_clusters)
    ]
    # Create the figure
    fig_width = 8 + (n_clusters // 4) * 2  # Dynamically adjust width based on the number of clusters
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    pdf_path = f'{path}/{pdf_filename}'
    with PdfPages(pdf_path) as pdf:
        # Plot the PCA results with transparency based on probabilities
        for i, name in enumerate(sample_names):
            cluster = cluster_assignments[i]
            probability = probabilities[i, cluster]
            
            # Normalize the probability for transparency: scale between min probability and 1
            min_prob = min_probabilities[cluster]
            if min_prob >= 0.98:
                normalized_prob = 1  # If all samples have probability 1, set to 1
            else:
                normalized_prob = (probability - min_prob) / (1 - min_prob)
            normalized_prob = max(0, min(normalized_prob, 1))  # Ensure it's within [0, 1]
            # Use the colormap for the cluster to determine the color
            cmap = cmap_dict[cluster]
            # Adjust for min_prob == 1
            if min_prob >= 0.98:
                color = colors[cluster]  # Use the cluster's base color directly
            else:
                color = cmap(normalized_prob)
            ax.scatter(pc1[i], pc2[i], color=color, edgecolor='k', s=20)
        ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)' if explained_variance is not None else 'PC1')
        ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)' if explained_variance is not None else 'PC2')
        ax.set_title('PCA Result Colour Coded\nBased on GMM Cluster Assignment')
        # Normalize for color mapping (from 0 to 1)
        norm = Normalize(vmin=0, vmax=1)
        # Add colorbars for each cluster using create_truncated_colormap
        cbar_width = 0.02
        cbar_height = 0.15
        cbar_spacing = 0.05
        for idx, cluster in enumerate(range(n_clusters)):
            col = idx // 4  # Calculate which column the colorbar should be in (0 for first column, 1 for second)
            row = idx % 4   # Calculate the row position within the column
            cbar_x = 0.75 + col * (cbar_width + 0.07)  # Adjust x position for column with more spacing
            cbar_y = 0.7 - row * (cbar_height + cbar_spacing)  # Adjust y position for row
            cmap = cmap_dict[cluster]
            cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
            # Get the minimum probability for the cluster
            min_prob = min_probabilities[cluster]
            # Adjust normalization and colorbar behavior
            if min_prob >= 0.98:
                # Create a solid color ScalarMappable with the same color for the entire bar
                solid_color = colors[cluster]  # Use the cluster's base color directly
                fig.colorbar(
                    ScalarMappable(norm=Normalize(vmin=1, vmax=1), cmap=LinearSegmentedColormap.from_list('solid', [solid_color, solid_color], N=2)),
                    cax=cbar_ax, orientation='vertical', ticks=[1]
                )
                cbar_ax.set_yticklabels(['1'])  # Set the label to '1' since it's a constant bar
            else:
                # Create and add the gradient colorbar for normal cases
                norm = Normalize(vmin=min_prob, vmax=1)
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar_ax, orientation='vertical', ticks=np.round(np.linspace(min_prob, 1, 6), 2)
                )
            
            cbar_ax.set_title(f'Cluster {cluster + 1}', fontsize=10)
        
        plt.subplots_adjust(right=0.7)  # Adjust space for legends with more space on the right
        ax.grid(True)
        pdf.savefig(fig)
        png_filename = f'{path}/PCA_gradient_{cluster + 1}.png'
        fig.savefig(png_filename, dpi=300)
        plt.close(fig)

def compute_normalized_distances(transformed_data, cluster_assignments, sample_names, n_clusters):
    """
    Compute normalized distances of samples from their assigned cluster centers.
    """
    cluster_centers = []
    for cluster in range(n_clusters):
        cluster_points = transformed_data[cluster_assignments == cluster]
        cluster_center = cluster_points.mean(axis=0)  # Compute centroid
        cluster_centers.append(cluster_center)

    distances = []
    max_distances = [0] * n_clusters  # Track max distance per cluster

    for i, cluster in enumerate(cluster_assignments):
        center = cluster_centers[cluster]
        distance = euclidean(transformed_data[i], center)
        distances.append(distance)
        max_distances[cluster] = max(max_distances[cluster], distance)

    # Normalize distances
    normalized_distances = [
        (distances[i] / max_distances[cluster]) if max_distances[cluster] != 0 else 0
        for i, cluster in enumerate(cluster_assignments)
    ]

    # Round distances and create report
    normalized_distances = np.round(normalized_distances, 4)
    distance_report = {name: norm_dist for name, norm_dist in zip(sample_names, normalized_distances)}

    return distance_report

#def perform_gmm(combined_matrix, n_components=2, max_iter=10000, n_init=2000, tol=1e-3, random_state=1, covariance_type='spherical', reg_covar=1e-3):
def perform_gmm(combined_matrix, n_components=2, max_iter=10000, n_init=2500, tol=1e-6, random_state=1, covariance_type='spherical', reg_covar=1e-3):
    """
    Perform Gaussian Mixture Modeling (GMM) on the combined matrix and return the probabilities of cluster assignment.
    """
    # min_vals and max_vals in log space
    min_vals = np.min(combined_matrix, axis=0)
    max_vals = np.max(combined_matrix, axis=0)
    # Convert min and max values from log space to linear space
    min_vals_lin = np.exp(min_vals)
    max_vals_lin = np.exp(max_vals)
    if n_components == 2:
        # Compute the arithmetic mean in linear space
        mean_1 = min_vals
        #mean_2 = max_vals  # Arithmetic mean in linear space, converted back to log
        mean_2 = np.log((min_vals_lin + max_vals_lin) / 2)  # Arithmetic mean in linear space, converted back to log
        means_init = np.vstack([mean_1, mean_2])
    elif n_components == 3:
        mean_1 = min_vals
        mean_2 = np.log((min_vals_lin + max_vals_lin) / 2)  # Arithmetic mean
        mean_3 = max_vals
        means_init = np.vstack([mean_1, mean_2, mean_3])
    elif n_components > 3:
        means_init = np.zeros((n_components, combined_matrix.shape[1]))
        means_init[0] = min_vals
        means_init[-1] = max_vals
        # Distribute other means evenly between min and max in linear space
        for i in range(1, n_components - 1):
            alpha = i / (n_components - 1)
            mean_lin = (1 - alpha) * min_vals_lin + alpha * max_vals_lin  # Linear interpolation in linear space
            means_init[i] = np.log(mean_lin)  # Convert back to log space
    else:
        raise ValueError("n_components must be greater than or equal to 2.")
    gmm_model = GaussianMixture(
        n_components=n_components, max_iter=max_iter, n_init=n_init,
        means_init=means_init, tol=tol,
        random_state=random_state, covariance_type=covariance_type, reg_covar=reg_covar)
    gmm_model.fit(combined_matrix)
    cluster_assignments = gmm_model.predict(combined_matrix)
    probabilities = gmm_model.predict_proba(combined_matrix)
    probabilities_rounded = np.round(probabilities, 4)
    row_sums = np.sum(probabilities_rounded, axis=1)
    adjustments = 1 - row_sums
    max_indices = np.argmax(probabilities_rounded, axis=1)  # Find the index of the highest value in each row
    probabilities_rounded[np.arange(probabilities_rounded.shape[0]), max_indices] += adjustments  # Add the difference to the highest probability in each row
    return gmm_model, cluster_assignments, probabilities_rounded


def extract_number_after_n(filename):
    # Split the string by '_n' and take the last part
    try:
        return int(filename.split('_n')[-1])  # Convert the last part to an integer
    except (ValueError, IndexError):
        return None  # Return None if the format is incorrect or conversion fails


def plot_cluster_probabilities_sorted_multicol(probabilities, sample_names, path, n_components=2, pdf_filename="cluster_report.pdf"):
    """
    Plot normalized horizontal bar plots for the probabilities of cluster assignments for each sample.
    """
    max_samples_per_column = 100
    columns_per_plot = 3
    max_samples_per_plot = max_samples_per_column * columns_per_plot
    assigned_clusters = np.argmax(probabilities, axis=1)
    sorted_indices = np.lexsort((-np.max(probabilities, axis=1), assigned_clusters))
    probabilities_sorted = probabilities[sorted_indices]
    sample_names_sorted = np.array(sample_names)[sorted_indices]
    #sample_mapping_nums = np.array([extract_number_after_n(name) for name in sample_names_sorted])
    #assigned_clusters_sorted = assigned_clusters[sorted_indices]
    sample_ids = np.arange(1, len(sample_names_sorted) + 1)
    sample_ids_str = [f'{id:0{len(str(len(sample_names)))}d}' for id in sample_ids]
    # Initialize PdfPages to save plots into a multi-page PDF
    with PdfPages(f'{path}/{pdf_filename}') as pdf:
        # Number of full plots needed (each with up to three columns)
        num_plots = (len(sample_names_sorted) + max_samples_per_plot - 1) // max_samples_per_plot
        for plot_idx in range(num_plots):
            start_idx = plot_idx * max_samples_per_plot
            end_idx = min(start_idx + max_samples_per_plot, len(sample_names_sorted))
            samples_on_plot = end_idx - start_idx
            #columns_on_plot = min(len(sample_names_sorted)//max_samples_per_column + 1, columns_per_plot)
            scaleSmpCol = 1
            figH = 12
            if samples_on_plot/max_samples_per_plot< 0.25:
                figH = 4
                scaleSmpCol = 0.25
            if samples_on_plot/max_samples_per_plot < 0.5:
                figH = 6
                scaleSmpCol = 0.5
            elif samples_on_plot/max_samples_per_plot < 0.75:
                figH = 9
                scaleSmpCol = 0.75
            columns_on_plot = min(samples_on_plot//int(max_samples_per_column*scaleSmpCol) + 1, columns_per_plot)
            sample_ids_on_plot = sample_ids_str[start_idx:end_idx]  # Top to bottom order now
            #print("len(sample_ids_on_plot)", len(sample_ids_on_plot))
            probabilities_on_plot = probabilities_sorted[start_idx:end_idx]  # Top to bottom order now
            cluster_probabilities = {i: [] for i in range(n_components)}
            for i in range(len(sample_ids_on_plot)):
                for component in range(n_components):
                    cluster_probabilities[component].append(probabilities_on_plot[i, component])
            figW = 8.5
            #figW = 5.5
            #figH = 8
            if plot_idx == num_plots-1:
                max_samples_per_column = samples_on_plot//columns_on_plot + samples_on_plot%columns_on_plot     
                # print("max_samples_per_column", max_samples_per_column)
                # print("columns_on_plot", columns_on_plot)
            figsizeFactor = 1
            # if samples_on_plot // max_samples_per_plot == 0:
            #     figsizeFactor = min(samples_on_plot/max_samples_per_column * figsizeFactor + 0.2, 1)
            # Create a new figure for each set of three columns (DINA4 format: 8.27 x 11.69 inches)
            fig, axs = plt.subplots(1, columns_on_plot, figsize=(figW, figsizeFactor*figH))  # DINA4 width split across 3 columns
            if columns_on_plot == 1:
                axs = [axs]
            bar_height = 0.65  # Height of the horizontal bars
            for col in range(columns_on_plot):
                col_start = col * max_samples_per_column
                col_end = min(col_start + max_samples_per_column, len(sample_ids_on_plot))
                bottom = np.zeros(col_end - col_start)
                
                # Instead of using sample_ids_on_plot (padded strings) as y-values,
                # create an array of numeric positions (0, 1, 2, ...)
                y_positions = np.arange(col_end - col_start)
                # But keep the padded labels for display, in the reversed order
                y_labels = sample_ids_on_plot[col_start:col_end][::-1]
                # Reverse probabilities for plotting (order must match y_positions)
                probabilities_for_plot = probabilities_on_plot[col_start:col_end][::-1]
                
                for component in range(n_components):
                    axs[col].barh(
                        y_positions,  # use numeric positions for plotting
                        probabilities_for_plot[:, component],  # probabilities for the current component
                        bar_height,
                        label=f'Cluster {component + 1}' if col == 0 else "",
                        left=bottom
                    )
                    bottom += probabilities_for_plot[:, component]
                    
                axs[col].set_xlabel('Probability of Cluster Membership')
                axs[col].set_ylabel('Samples (by ID)')
                axs[col].tick_params(axis='y', labelsize=7)
                # Set numeric y-ticks, then assign your padded labels
                axs[col].set_yticks(y_positions)
                axs[col].set_yticklabels(y_labels)
                axs[col].set_ylim(-0.2, len(y_positions))
                axs[col].set_xlim(0, 1)
            # Add a single title across the figure
            #fig.suptitle('Membership Probabilities per Sample', fontsize=10)
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=n_components, bbox_to_anchor=(0.5, 0.99))
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Reduced rect to leave less space at the top
            pdf.savefig(fig)
            png_filename = f'{path}/cluster_probabilities_plot_{plot_idx + 1}.png'
            fig.savefig(png_filename, dpi=300)
            plt.close()


def plot_weighted_profiles(probabilities, sample_names, combined_matrix, plot_path, reverse, method, n_components, pdf_filename="weighted_profiles.pdf", libtype="paired"):
    """
    Calculate and plot weighted representative damage profiles based on the cluster probabilities,
    and save the plots to a multi-page PDF.
    
    Parameters:
    - probabilities: A 2D array where each row corresponds to a sample and each column to a cluster.
    - sample_names: A list of sample names.
    - combined_matrix: A 2D array where each row contains concatenated 5p and 3p data for a sample.
    - plot_path: The path where the plots and PDF will be saved.
    - reverse: A boolean indicating whether the 3p data should be reversed.
    - method: A string representing the clustering method.
    - n_components: The number of clusters.
    - pdf_filename: The name of the output PDF file.
    """
    pdf_path = f'{plot_path}/{pdf_filename}'

    global_max_damage = 0
    for i, sample_name in enumerate(sample_names):
        combined_data = combined_matrix[i]
        max_sample_damage = np.max(combined_data)  # Find max value in this sample's damage profile
        global_max_damage = max(global_max_damage, max_sample_damage)
    global_max_damage += 0.1

    with PdfPages(pdf_path) as pdf:

        best_clusters = np.argmax(probabilities, axis=1)

        for cluster in range(n_components):
            # Initialize combined data dictionary with scalar zeros
            weighted_combined_data = {i: 0.0 for i in range(10)}
            cluster_combined_data = []
            total_prob = 0.0

            for i, sample_name in enumerate(sample_names):
                prob = probabilities[i, cluster]
                if prob <= 0:
                    continue

                # pull out your 10‐element vector
                combined_data = combined_matrix[i]

                # only append to the *hard* cluster
                if best_clusters[i] == cluster:
                    cluster_combined_data.append(combined_data)

                # split, maybe reverse, concat, and weight exactly as before
                fivep_data  = combined_data[:5]
                threep_data = combined_data[5:]
                if not reverse:
                    threep_data = threep_data[::-1]
                reordered = np.concatenate([fivep_data, threep_data])

                for pos in range(10):
                    weighted_combined_data[pos] += reordered[pos] * prob
                total_prob += prob
            
            # Normalize by total probability
            for pos in range(10):
                if total_prob > 0:
                    weighted_combined_data[pos] /= total_prob
            
            # Plotting the results
            save_weighted_profiles_to_tsv(weighted_combined_data, plot_path, f'cluster_{cluster + 1}_weighted_profile')
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            plot_prof_substitutions(axs[0], weighted_combined_data, global_max_damage, cluster_combined_data, substitution_type='C>T', color='red', positions_range=(0, 5), xlabel="Position from 5' end", libtype=libtype)

            subtype_id = 'G>A'
            if libtype == "mixed":
                subtype_id = 'C>T / G>A'
            plot_prof_substitutions(axs[1], weighted_combined_data, global_max_damage, cluster_combined_data, substitution_type=subtype_id, color='blue', positions_range=(5, 10), xlabel="Position from 3' end", libtype=libtype)
            plt.suptitle(f'Representative Substitution Frequencies for Cluster {cluster + 1} ({method})')
            plt.tight_layout()
            pdf.savefig(fig)
            png_filename = f'{plot_path}/cluster_{cluster + 1}_weighted_profiles.png'
            fig.savefig(png_filename, dpi=300)
            plt.close(fig)
    logger.info(f"Weighted profiles saved to {pdf_path}")



def plot_all_profiles(sample_names, combined_matrix, plot_path, reverse, libtype):
    """
    Plot each sample’s 5' and 3' damage profiles as a two-panel figure,
    and save each one as an individual PNG.

    Parameters:
    - sample_names: List of sample name strings.
    - combined_matrix: 2D array [n_samples × 10] of damage values.
    - plot_path: Directory where PNGs will be written.
    - reverse: If False, reverse the 3' portion before plotting.
    - single_strand: Passed through to your plotting helper.
    """
    os.makedirs(plot_path, exist_ok=True)

    # if libtype=="mixed":
    #     logger.info(f"Warning: Mixed libraries. Plots will be 5p C>T and 3p G>A.")

    # shared y-axis limit
    global_max = np.max(combined_matrix)
    y_lim = global_max + 0.1

    for i, name in enumerate(sample_names):
        data = combined_matrix[i]
        fivep = data[:5]
        threep = data[5:]
        if not reverse:
            threep = threep[::-1]
        reordered = np.concatenate([fivep, threep])

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        # 5' panel: C>T
        plot_prof_substitutions(
            axes[0], reordered, y_lim, [],
            substitution_type='C>T',
            color='red',
            positions_range=(0, 5),
            xlabel="Position from 5' end",
            libtype=libtype
        )
        # 3' panel: G>A

        subtype_id = 'G>A'
        if libtype == "mixed":
            subtype_id = 'C>T / G>A'

        plot_prof_substitutions(
            axes[1], reordered, y_lim, [],
            substitution_type=subtype_id,
            color='blue',
            positions_range=(5, 10),
            xlabel="Position from 3' end",
            libtype=libtype
        )

        fig.suptitle(f"{name}", fontsize=10)
        plt.tight_layout()

        out_png = os.path.join(plot_path, f"{name}_profile.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

    logger.info(f"All profiles saved as PNGs in {plot_path}")



def plot_prof_substitutions(ax, all_combined_data, max_dam, cluster_combined_data, substitution_type='C>T', color='red', positions_range=(0, 5), xlabel="Position from 5' end", libtype="paired"):
    """
    Create a scatter plot with a continuous line for substitutions,
    and overlay cluster-specific profiles as thin grey lines.
    """
    positions = list(range(*positions_range))
    # set up x-axis labels
    if substitution_type == 'G>A' or substitution_type == 'C>T / G>A':
        x_labels = [-4, -3, -2, -1, 0]
    else:
        x_labels = positions

    substitution_type_label=substitution_type
    if libtype == "single":
        substitution_type_label='C>T'

    # first, plot each cluster in light grey
    if cluster_combined_data is not None:
        for cluster in cluster_combined_data:
            # extract only the relevant slice for this range
            cluster_freqs = [cluster[pos] for pos in positions]
            ax.plot(positions, cluster_freqs, color='grey', linewidth=0.5, linestyle='-', alpha=0.6, zorder=1)

    # then plot the main aggregated curve on top
    frequencies = [all_combined_data[pos] for pos in positions]
    ax.plot(positions, frequencies, color=color, linestyle='-', marker='o', zorder=2)

    # labels, limits, ticks, grid
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(f'{substitution_type_label} Substitution Frequency', fontsize=10)
    ax.set_ylim(0, max_dam)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.grid(axis='y', linestyle='--')


def save_weighted_profiles_to_tsv(weighted_combined_data, output_dir, prefix="none"):
    """
    Save the weighted combined data into two TSV files for the 5' and 3' ends.
    
    Parameters:
    - weighted_combined_data: A dictionary containing weighted substitution frequencies
                              for positions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
                              Index 0-4 correspond to positions 1-5, and 5-9 to positions -5 to -1.
    - output_dir: The directory where the TSV files will be saved.
    - prefix: The prefix for the output file names (default: "ancientGut_dhigh").
    """
    # Column headers for the TSV files
    substitution_columns = [
        "A>C", "A>G", "A>T", "C>A", "C>G", "C>T", 
        "G>A", "G>C", "G>T", "T>A", "T>C", "T>G"
    ]
    
    # Initialize dataframes with zeroed substitution columns as floats
    data_5p = pd.DataFrame(0.0, index=range(5), columns=substitution_columns)
    data_3p = pd.DataFrame(0.0, index=range(5), columns=substitution_columns)
    
    # Fill the relevant columns with weighted frequencies
    for i in range(5):  # 5' positions: 0, 1, 2, 3, 4
        data_5p.at[i, "C>T"] = round(weighted_combined_data[i], 3)  # Populate data_5p with 5' data
        
    for i, pos in enumerate(reversed(range(5, 10))):  # 3' positions: 5, 6, 7, 8, 9
        data_3p.at[i, "G>A"] = round(weighted_combined_data[pos], 3)  # Populate data_3p with 3' data
    
    # Save to TSV files
    tsv_5p_path = f"{output_dir}/{prefix}_5p.prof"
    tsv_3p_path = f"{output_dir}/{prefix}_3p.prof"
    data_5p.to_csv(tsv_5p_path, sep="\t", index=False)
    data_3p.to_csv(tsv_3p_path, sep="\t", index=False)


def validate_concatenated_row(concatenated_row):
    return True


def get_strand_columns( matrix_5p, matrix_3p, column_5p, column_3p, libtype):
    """
    Extracts 5' and 3' strand data according to the requested mode.

    Parameters
    ----------
    matrix_5p : pd.DataFrame
        DataFrame containing the 5' data.
    matrix_3p : pd.DataFrame
        DataFrame containing the 3' data.
    column_5p : int
        1-based index of the column to take from each DataFrame for the 5' strand.
    column_3p : Optional[int], default=None
        1-based index of the column to take for the 3' strand when not single_strand.
        If None, defaults to the same as column_5p.
    fullon : bool, default=False
        If True, concatenate the selected columns from both 5p and 3p matrices.
    single_strand : bool, default=False
        If True, both 5' and 3' strands come from the same column index.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (five_prime_data, three_prime_data)
    """
    # Convert to zero-based indices
    idx5 = column_5p - 1
    idx3 = idx5 if libtype=="single" else (column_3p - 1 if column_3p else idx5)

    # Grab raw arrays
    arr5p = matrix_5p.iloc[:, idx5].values
    arr3p = matrix_3p.iloc[:, idx3].values

    if libtype=="mixed":
        # If mixed, also grab the complementary half from the other matrix
        other5 = matrix_3p.iloc[:, idx5].values
        other3 = matrix_5p.iloc[:, idx3].values
        five_prime = np.concatenate([arr5p, other5[::-1]])
        three_prime = np.concatenate([other3, arr3p[::-1]])
    else:
        five_prime = arr5p
        three_prime = arr3p[::-1]

    return five_prime, three_prime



def process_directory_combined(directory, num_rows_5p, num_rows_3p, minmap = 1000, column_5p=6, column_3p=7, libtype="paired"):
    """
    Process all matrices in the directory and build a combined matrix for analysis.
    """
    # Load all files once
    matrix_files = glob.glob(os.path.join(directory, '*.prof'))
    # # Extract basenames and classify them
    basenames = set(os.path.basename(f).rsplit('_', 1)[0] for f in matrix_files)
    # Create dictionaries to store loaded matrices
    matrix_5p_dict = {}
    matrix_3p_dict = {}
    # Pre-load all matrices into dictionaries
    for file_path in matrix_files:
        basename = os.path.basename(file_path).rsplit('_', 1)[0]
        if basename not in basenames:
            continue
        if file_path.endswith("3p.prof"):
            matrix_3p_dict[basename] = load_matrix(file_path)
        elif file_path.endswith("5p.prof"):
            matrix_5p_dict[basename] = load_matrix(file_path)
    combined_matrix = []
    sample_names = []
    sample_names_outsourced = []
    # Iterate over the filtered basenames
    for basename in basenames:
        # Get the matrices from the preloaded dictionaries
        matrix_3p = matrix_3p_dict.get(basename)
        matrix_5p = matrix_5p_dict.get(basename)
        # Check if matrices are loaded correctly and valid
        if matrix_3p is None or matrix_5p is None or check_for_nan(matrix_3p) or check_for_nan(matrix_5p):
            continue

        five_prime_data, three_prime_data = get_strand_columns(matrix_5p, matrix_3p, column_5p=column_5p, column_3p=column_3p, libtype=libtype)

        #concatenated_row = np.concatenate([column_5p_data, column_3p_data[::-1]])
        concatenated_row = np.concatenate([five_prime_data, three_prime_data])
        # Determine the number of mapped reads
        nMapped = int(basename.split('_n')[-1])
        # Add the concatenated row based on conditions
        if nMapped >= minmap and validate_concatenated_row(concatenated_row):
                combined_matrix.append(concatenated_row)
                sample_names.append(basename)
        else:
            sample_names_outsourced.append(basename)
    # Convert the combined matrix to a NumPy array for further processing
    combined_matrix = np.array(combined_matrix)
    #print(f"Combined matrix shape: {combined_matrix.shape}")
    return combined_matrix, sample_names, sample_names_outsourced


def main():

    print_package_versions()

    parser = argparse.ArgumentParser(description='Cluster and plot damage profiles.')
    parser.add_argument('-i', type=str, required=True, metavar='INPUT_DIR',
                        help='Path to the directory containing the profiles generated with bam2prof.')
    parser.add_argument('-o', type=str, required=True, metavar='OUTPUT_DIR',
                        help='Path where your plots will be saved.')
    parser.add_argument('-k', type=int, default=4, metavar='Clusters',
                        help='Run clustering from 2 to k. (default: %(default)s)')
    parser.add_argument('-q', type=int, default=0, metavar='Less Plots (faster)',
                        help='Do not plot probability per sample. Write to TSV only. [off=0,on=1](default: %(default)s)')
    parser.add_argument('-m', type=int, default=1000, metavar='Minimum Mapped Reads',
                        help='Require at least m reads to be mapped to a reference to be included in clustering (default: %(default)s)')
    parser.add_argument('-single', type=int, default=0, metavar='ss-Library only',
                        help='Set to 1 if reads are single strand library (default: %(default)s)')
    parser.add_argument('-plotall', type=int, default=0, metavar='Plot All Profs',
                        help='Set to 1 if all profs should be plotted individually. Slow! (default: %(default)s)')
    parser.add_argument('-mixed', type=int, default=0, metavar='ss- and paired-end Input',
                        help='Set to 1 if input are bam files from ss- and paired-end libraries. (default: %(default)s)')
    parser.add_argument('-lib', choices=['single', 'paired', 'mixed'], default='paired', help=('Type of library reads in your input BAMs. (default: %(default)s)')
    )


    args = parser.parse_args()
    input_dir = args.i
    plot_path = args.o
    cluster_k = args.k
    less_plots = args.q
    min_map = args.m
    #single_strand_flag = args.single
    all_plots_flag = args.plotall
    libtype = args.lib

    # Create the output directory if it doesn't exist
    os.makedirs(plot_path, exist_ok=True)

    now = datetime.datetime.now()
    logger.info("Preparing Matrix: %s", now)

    # get fullon matrix
    combined_matrix_compute, sample_names, sample_names_outsourced = process_directory_combined(
        input_dir,
        num_rows_5p=5,
        num_rows_3p=5,
        minmap = min_map,
        column_5p=6,
        column_3p=7,
        libtype=libtype,
    )

    combined_matrix  =  np.hstack((combined_matrix_compute[:, :5], combined_matrix_compute[:, -5:]))

    now = datetime.datetime.now()
    logger.info("Preparing Matrix Done %s", now)

    # # Mask out zeros by replacing them with np.inf (so they are ignored in the min calculation)
    masked_matrix = np.where(combined_matrix_compute == 0, np.inf, combined_matrix_compute)
    # # Find the minimum value of the non-zero elements
    min_non_zero_value = np.min(masked_matrix)
    small_constant = min_non_zero_value
    matrix_safe = np.where(combined_matrix_compute == 0, small_constant, combined_matrix_compute)

    # Apply the natural logarithm 
    combined_matrix_compute = np.log(matrix_safe)
    combined_matrix_compute_pca  =  np.hstack((combined_matrix_compute[:, :5], combined_matrix_compute[:, -5:]))

    # print(combined_matrix_compute)
    # print(combined_matrix_compute_pca)
    # print(combined_matrix)

    # Save file with outsourced sample ids
    outsource_file = os.path.join(plot_path, 'outsourced_samples.tsv')
    df = pd.DataFrame(sample_names_outsourced, columns=['SampleName'])
    df.to_csv(outsource_file, sep='\t', index=False)

    # Iterate over k_iter for GMM clustering
    for k_iter in range(2, cluster_k+1):

        data_rows, data_cols = combined_matrix.shape
        if k_iter > data_rows:
            logger.info(f'n_samples ({data_rows}) < k ({k_iter}). Skipping...')
            continue
        logger.info(f'Running clustering for k = {k_iter}...')
        pdf_list = []
        
        # Perform GMM clustering
        # if only5p_flag != 1:
        gmm_model, gmm_cluster_assignments, gmm_probabilities = perform_gmm(combined_matrix_compute, n_components=k_iter)
        # else:
        #     gmm_model, gmm_cluster_assignments, gmm_probabilities = perform_gmm(combined_matrix_only5p, n_components=k_iter)
        now = datetime.datetime.now()
        logger.info("Clustering Done %s", now)

        # Make PCA Plot with gradient
        # transformed_data, explained_variance, pca = perform_pca(combined_matrix, None)
        # if only5p_flag != 1:
        transformed_data, explained_variance, pca = perform_pca(combined_matrix_compute_pca, None)
        # else:
        #     transformed_data, explained_variance, pca = perform_pca(combined_matrix_only5p, None)
        pca_plot_path = create_plot_directories(plot_path, "PCA", k_iter)
        n_clu = gmm_probabilities.shape[1]
        cluster_assignments = np.argmax(gmm_probabilities, axis=1)
        distances = compute_normalized_distances(transformed_data, cluster_assignments, sample_names, n_clu)
        now = datetime.datetime.now()
        logger.info("PCA Done %s", now)
        
        if less_plots == 0:
            plot_pca_gradient(
                transformed_data, sample_names, gmm_probabilities, pca_plot_path,
                explained_variance=explained_variance, pdf_filename=f"pca_k{k_iter}.pdf")
            pdf1 = os.path.join(pca_plot_path, f"pca_k{k_iter}.pdf")
            pdf_list.append(pdf1)

        gmm_plot_path = create_plot_directories(plot_path, "GMM", k_iter)   
        if less_plots == 0: 
            plot_cluster_probabilities_sorted_multicol(
                gmm_probabilities, sample_names, gmm_plot_path, k_iter,
                pdf_filename=f"cluster_report_k{k_iter}.pdf")
            pdf2 = os.path.join(gmm_plot_path, f"cluster_report_k{k_iter}.pdf")
            pdf_list.append(pdf2)

        if all_plots_flag == 1:
            all_plots_path= os.path.join(plot_path, "plots_all_profs")
            plot_all_profiles(sample_names, combined_matrix, all_plots_path, True, libtype=libtype)

        
        plot_weighted_profiles(
            gmm_probabilities, sample_names, combined_matrix, gmm_plot_path, True,
            "GMM", k_iter, pdf_filename=f"weighted_profiles_k{k_iter}.pdf",  libtype=libtype)
        pdf3 = os.path.join(gmm_plot_path, f"weighted_profiles_k{k_iter}.pdf")
        pdf_list.append(pdf3)

        now = datetime.datetime.now()
        logger.info("Plotting Done %s", now)

        save_probs_ids_tsv(gmm_probabilities, sample_names, distances, gmm_plot_path, k_iter, filename_prefix="cluster_report_k")
        now = datetime.datetime.now()
        logger.info("Export TSV Done %s", now)
    
        #pdf_list = [pdf1, pdf2, pdf3]
        
        # Merge the PDFs into a single file
        merged_pdf_path = os.path.join(plot_path, f"damage_report_k{k_iter}.pdf")
        merge_pdfs(pdf_list, merged_pdf_path)

        # Scale the final merged PDF to match the width of the cluster report
        target_width = 8.5 * 72  # 72 points per inch
        scaled_pdf_path = os.path.join(plot_path, f"damage_report_k{k_iter}.pdf")
        scale_pdf(merged_pdf_path, scaled_pdf_path, target_width)
        
        logger.info(f"PDFs merged and scaled to {target_width/72:.2f} inches width. Final PDF: {scaled_pdf_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
        sys.exit(0)
