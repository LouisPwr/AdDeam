#!/usr/bin/env python
# coding: utf-8

import os
import random
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from PyPDF2 import PdfReader, PdfWriter

# ### Helper Functions ###

def check_for_nan(matrix):
    """Check if the matrix contains any NaN values."""
    return matrix.isnull().values.any() or np.any(np.isnan(matrix.values))

def load_matrix(file_path):
    """Load the matrix from a file."""
    return pd.read_csv(file_path, sep='\s+')

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

def save_probs_ids_tsv(probabilities, sample_names, sample_ids, plot_path, filename="cluster_probabilities_with_ids.tsv"):
    """
    Save the probabilities matrix to a TSV file, where each row corresponds to a sample name and ID,
    and each column represents a cluster, with values as the cluster probabilities.
    """
    n_clusters = probabilities.shape[1]
    cluster_columns = [f'Cluster{i+1}' for i in range(n_clusters)]
    df = pd.DataFrame(probabilities, columns=cluster_columns)
    df.insert(0, 'SampleID', sample_ids)
    df.insert(1, 'SampleName', sample_names)
    tsv_path = f'{plot_path}/{filename}'
    df.to_csv(tsv_path, sep='\t', index=False)
    print(f"Cluster probabilities with IDs saved to {tsv_path}")

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

def plot_pca_gradient(transformed_data, sample_names, probabilities, path='.', explained_variance=None, pdf_filename="pca_gradient_plots.pdf"):
    """
    Plot PCA results with color gradient based on the highest cluster probability per sample,
    and save the plot to a PDF.
    """
    n_clusters = probabilities.shape[1]
    colors = plt.get_cmap('tab10').colors[:n_clusters]
    cluster_assignments = np.argmax(probabilities, axis=1)
    pc1 = transformed_data[:, 0]
    pc2 = transformed_data[:, 1]
    pdf_path = f'{path}/{pdf_filename}'
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(sample_names):
            cluster = cluster_assignments[i]
            color = colors[cluster]
            ax.scatter(pc1[i], pc2[i], color=color, edgecolor='k', s=30)
        ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)' if explained_variance is not None else 'PC1')
        ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)' if explained_variance is not None else 'PC2')
        ax.set_title('PCA Result Colour Coded\nBased on GMM Cluster Assignment')
        norm = Normalize(vmin=0, vmax=1)
        for cluster in range(n_clusters):
            cmap = create_truncated_colormap(colors[cluster])
            cbar_ax = fig.add_axes([0.85, 0.7 - cluster * 0.2, 0.03, 0.15])
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical', ticks=np.arange(0, 1.2, 0.2))
            cbar_ax.set_title(f'Cluster {cluster + 1}', fontsize=10)
        plt.subplots_adjust(right=0.8)
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"PCA gradient plot saved to {pdf_path}")

def perform_gmm(combined_matrix, n_components=2, max_iter=1000, n_init=1000, init_params='random', tol=1e-3, random_state=1):
    """
    Perform Gaussian Mixture Modeling (GMM) on the combined matrix and return the probabilities of cluster assignment.
    """
    min_vals = np.min(combined_matrix, axis=0)
    max_vals = np.max(combined_matrix, axis=0)
    if n_components == 2:
        mean_1 = min_vals
        mean_2 = (min_vals + max_vals) / 2
        means_init = np.vstack([mean_1, mean_2])
    elif n_components == 3:
        mean_1 = min_vals
        mean_2 = (min_vals + max_vals) / 2
        mean_3 = max_vals
        means_init = np.vstack([mean_1, mean_2, mean_3])
    elif n_components > 3:
        means_init = np.zeros((n_components, combined_matrix.shape[1]))
        means_init[0] = min_vals
        means_init[-1] = max_vals
        for i in range(1, n_components - 1):
            alpha = i / (n_components - 1)
            means_init[i] = min_vals + alpha * (max_vals - min_vals)
    else:
        raise ValueError("n_components must be greater than or equal to 2.")
    gmm_model = GaussianMixture(
        n_components=n_components, max_iter=max_iter, n_init=n_init,
        means_init=means_init, init_params=init_params, tol=tol,
        random_state=random_state, covariance_type='diag', reg_covar=1e-3)
    gmm_model.fit(combined_matrix)
    cluster_assignments = gmm_model.predict(combined_matrix)
    probabilities = gmm_model.predict_proba(combined_matrix)
    return gmm_model, cluster_assignments, probabilities

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
    #assigned_clusters_sorted = assigned_clusters[sorted_indices]
    sample_ids = np.arange(1, len(sample_names_sorted) + 1)
    sample_ids_str = [f'{id:0{len(str(len(sample_names)))}d}' for id in sample_ids]
    # Save probabilities, sample names, and IDs to a TSV file
    save_probs_ids_tsv(probabilities_sorted, sample_names_sorted, sample_ids_str, path)
    # Initialize PdfPages to save plots into a multi-page PDF
    with PdfPages(f'{path}/{pdf_filename}') as pdf:
        # Number of full plots needed (each with up to three columns)
        num_plots = (len(sample_names_sorted) + max_samples_per_plot - 1) // max_samples_per_plot
        for plot_idx in range(num_plots):
            start_idx = plot_idx * max_samples_per_plot
            end_idx = min(start_idx + max_samples_per_plot, len(sample_names_sorted))
            samples_on_plot = end_idx - start_idx
            columns_on_plot = min(len(sample_names_sorted)//max_samples_per_column + 1, columns_per_plot)
            sample_ids_on_plot = sample_ids_str[start_idx:end_idx]  # Top to bottom order now
            probabilities_on_plot = probabilities_sorted[start_idx:end_idx]  # Top to bottom order now
            cluster_probabilities = {i: [] for i in range(n_components)}
            for i in range(len(sample_ids_on_plot)):
                for component in range(n_components):
                    cluster_probabilities[component].append(probabilities_on_plot[i, component])
            figW = 8.5
            figH = 12
            figsizeFactor = 1
            if samples_on_plot // max_samples_per_plot == 0:
                figsizeFactor = min(samples_on_plot/max_samples_per_column * figsizeFactor + 0.2, 1)
            # Create a new figure for each set of three columns (DINA4 format: 8.27 x 11.69 inches)
            fig, axs = plt.subplots(1, columns_on_plot, figsize=(figW, figsizeFactor*figH))  # DINA4 width split across 3 columns
            if columns_on_plot == 1:
                axs = [axs]
            bar_height = 0.65  # Height of the horizontal bars
            for col in range(columns_on_plot):
                col_start = col * max_samples_per_column
                col_end = min(col_start + max_samples_per_column, len(sample_ids_on_plot))
                bottom = np.zeros(col_end - col_start)
                # Reverse the sample IDs and the corresponding probabilities for plotting top to bottom
                sample_ids_for_plot = sample_ids_on_plot[col_start:col_end][::-1]
                probabilities_for_plot = probabilities_on_plot[col_start:col_end][::-1]  # Reverse probabilities
                for component in range(n_components):
                    axs[col].barh(
                        sample_ids_for_plot,  # Reversed sample IDs to plot top-to-bottom
                        probabilities_for_plot[:, component],  # Reversed probabilities for the current component
                        bar_height,
                        label=f'Cluster {component + 1}' if col == 0 else "",
                        left=bottom
                    )
                    bottom += probabilities_for_plot[:, component]
                axs[col].set_xlabel('Probability of Cluster Membership')
                axs[col].set_ylabel('Samples (by ID)')
                axs[col].tick_params(axis='y', labelsize=7)  # Adjust the '6' to your preferred font size
                axs[col].set_ylim(-0.2, len(sample_ids_for_plot))  # Ensures no excess space at the top or bottom
                axs[col].set_xlim(0, 1)  # Ensures no excess space at the top or bottom
            # Add a single title across the figure
            #fig.suptitle('Membership Probabilities per Sample', fontsize=10)
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=n_components, bbox_to_anchor=(0.5, 0.99))
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Reduced rect to leave less space at the top
            pdf.savefig(fig)
            plt.close()
    with open(f'{path}/sample_legend_k{n_components}.txt', 'w') as f:
        for i, sample_name in enumerate(sample_names_sorted):
            f.write(f'{sample_ids[i]:0{len(str(len(sample_names_sorted)))}d}: {sample_name}\n')

def find_matching_files(damage_dir, sample_names):
    file_dict = {}
    for sample_name in sample_names:
        file_pattern = f'{damage_dir}/*{sample_name}*.prof'
        matching_files = glob.glob(file_pattern)
        fivep_file = next((f for f in matching_files if '_5p.prof' in f), None)
        threep_file = next((f for f in matching_files if '_3p.prof' in f), None)
        if fivep_file and threep_file:
            file_dict[sample_name] = {'5p': fivep_file, '3p': threep_file}
    return file_dict

# def plot_weighted_profiles(probabilities, sample_names, damage_paths, plot_path, reverse, method, n_components, pdf_filename="weighted_profiles.pdf"):
#     """
#     Calculate and plot weighted representative damage profiles based on the cluster probabilities,
#     and save the plots to a multi-page PDF.
#     """
#     pdf_path = f'{plot_path}/{pdf_filename}'
#     with PdfPages(pdf_path) as pdf:
#         for cluster in range(n_components):
#             print("cluster", cluster)
#             weighted_combined_data = {i: [] for i in range(10)}
#             total_prob = 0
#             for i, sample_name in enumerate(sample_names):
#                 print(i, sample_name)
#                 prob = probabilities[i, cluster]
#                 if prob > 0:
#                     file_dict = find_matching_files(damage_paths, [sample_name])
#                     if sample_name not in file_dict:
#                         continue
#                     fivep_file = file_dict[sample_name]['5p']
#                     threep_file = file_dict[sample_name]['3p']
#                     fivep_data = pd.read_csv(fivep_file, delimiter='\t')["C>T"]
#                     threep_data = pd.read_csv(threep_file, delimiter='\t')["G>A"]
#                     if reverse:
#                         threep_data_reversed = threep_data.iloc[::-1].reset_index(drop=True)
#                     else:
#                         threep_data_reversed = threep_data
#                     combined_data = np.concatenate([fivep_data, threep_data_reversed])
#                     for pos in range(10):
#                         print("pos", pos)
#                         print(type(weighted_combined_data[pos]))
#                         if len(weighted_combined_data[pos]) == 0:
#                             weighted_combined_data[pos] = combined_data[pos] * prob
#                         else:
#                             weighted_combined_data[pos] += combined_data[pos] * prob
#                     total_prob += prob
#             for pos in range(10):
#                 weighted_combined_data[pos] /= total_prob
#             fig, axs = plt.subplots(1, 2, figsize=(8, 3))
#             plot_prof_substitutions(axs[0], weighted_combined_data, substitution_type='C>T', color='red', positions_range=(0, 5), xlabel="Position from 5' end")
#             plot_prof_substitutions(axs[1], weighted_combined_data, substitution_type='G>A', color='blue', positions_range=(5, 10), xlabel="Position from 3' end")
#             plt.suptitle(f'Representative Substitution Frequencies for Cluster {cluster + 1} ({method})')
#             plt.tight_layout()
#             pdf.savefig(fig)
#             plt.close(fig)
#     print(f"Weighted profiles saved to {pdf_path}")

def plot_weighted_profiles(probabilities, sample_names, damage_paths, plot_path, reverse, method, n_components, pdf_filename="weighted_profiles.pdf"):
    """
    Calculate and plot weighted representative damage profiles based on the cluster probabilities,
    and save the plots to a multi-page PDF.
    """
    pdf_path = f'{plot_path}/{pdf_filename}'
    with PdfPages(pdf_path) as pdf:
        for cluster in range(n_components):
            # Initialize combined data dictionary with arrays of zeros instead of lists
            weighted_combined_data = {i: np.zeros(10) for i in range(10)}
            total_prob = 0
            for i, sample_name in enumerate(sample_names):
                prob = probabilities[i, cluster]
                if prob > 0:
                    file_dict = find_matching_files(damage_paths, [sample_name])
                    if sample_name not in file_dict:
                        continue
                    fivep_file = file_dict[sample_name]['5p']
                    threep_file = file_dict[sample_name]['3p']
                    fivep_data = pd.read_csv(fivep_file, delimiter='\t')["C>T"]
                    threep_data = pd.read_csv(threep_file, delimiter='\t')["G>A"]
                    if reverse:
                        threep_data_reversed = threep_data.iloc[::-1].reset_index(drop=True)
                    else:
                        threep_data_reversed = threep_data
                    combined_data = np.concatenate([fivep_data, threep_data_reversed])
                    
                    # Weight and accumulate the combined data using arrays
                    for pos in range(10):
                        weighted_combined_data[pos] += combined_data[pos] * prob
                    total_prob += prob
            
            # Normalize by total probability
            for pos in range(10):
                if total_prob > 0:
                    weighted_combined_data[pos] /= total_prob
            
            # Plotting the results
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            plot_prof_substitutions(axs[0], weighted_combined_data, substitution_type='C>T', color='red', positions_range=(0, 5), xlabel="Position from 5' end")
            plot_prof_substitutions(axs[1], weighted_combined_data, substitution_type='G>A', color='blue', positions_range=(5, 10), xlabel="Position from 3' end")
            plt.suptitle(f'Representative Substitution Frequencies for Cluster {cluster + 1} ({method})')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Weighted profiles saved to {pdf_path}")


def plot_prof_substitutions(ax, all_combined_data, substitution_type='C>T', color='red', positions_range=(5, 10), xlabel="Position from 5' end"):
    """
    Create a scatter plot with a continuous line for substitutions.
    """
    positions = list(range(*positions_range))
    if substitution_type == 'G>A':
        x_labels = [-4, -3, -2, -1, 0]
    else:
        x_labels = positions
    frequencies = [all_combined_data[pos] for pos in positions]
    ax.plot(positions, frequencies, color=color, linestyle='-', marker='o')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(f'{substitution_type} Substitution Frequency', fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.grid(axis='y', linestyle='--')

def process_directory_combined(directory, num_rows_5p, num_rows_3p, num_columns, column_5p=6, column_3p=7, balance=[0,0,0]):
    """
    Process all matrices in the directory and build a combined matrix for analysis.
    """
    matrix_filesraw = glob.glob(os.path.join(directory, '*.prof'))
    matrix_files = matrix_filesraw
    basenames = set(os.path.basename(f).rsplit('_', 1)[0] for f in matrix_files)
    no_basenames = [basename for basename in basenames if "dnone" in basename]
    no_remove = int(len(no_basenames) * balance[0])
    no_base_rm = random.sample(no_basenames, no_remove)
    mid_basenames = [basename for basename in basenames if "dmid" in basename]
    mid_remove = int(len(mid_basenames) * balance[1])
    mid_base_rm = random.sample(mid_basenames, mid_remove)
    high_basenames = [basename for basename in basenames if "dhigh" in basename]
    high_remove = int(len(high_basenames) * balance[2])
    high_base_rm = random.sample(high_basenames, high_remove)
    basenames = basenames - set(high_base_rm) - set(mid_base_rm) - set(no_base_rm)
    combined_matrix = []
    sample_names = []
    for basename in basenames:
        files = [f for f in matrix_files if basename in f]
        try:
            matrix_3p = load_matrix(next(f for f in files if '3p' in f))
            matrix_5p = load_matrix(next(f for f in files if '5p' in f))
        except StopIteration:
            continue
        if check_for_nan(matrix_3p) or check_for_nan(matrix_5p):
            continue
        column_5p_data = matrix_5p.iloc[:, column_5p-1].values
        column_3p_data = matrix_3p.iloc[:, column_3p-1].values
        concatenated_row = np.concatenate([column_5p_data, column_3p_data[::-1]])
        combined_matrix.append(concatenated_row)
        sample_names.append(basename)
    combined_matrix = np.array(combined_matrix)
    print(f"Combined matrix shape: {combined_matrix.shape}")
    return combined_matrix, sample_names

def main():
    parser = argparse.ArgumentParser(description='Process and plot damage profiles.')
    parser.add_argument('-i', type=str, required=True,
                        help='Path to the directory containing the profiles generated with bam2prof.')
    parser.add_argument('-o', type=str, required=True,
                        help='Path where your plots will be saved.')
    args = parser.parse_args()

    input_dir = args.i
    plot_path = args.o

    # Create the output directory if it doesn't exist
    os.makedirs(plot_path, exist_ok=True)

    # Process all matrices and get results
    combined_matrix, sample_names = process_directory_combined(
        input_dir,
        num_rows_5p=5,
        num_rows_3p=5,
        num_columns=1,
        column_5p=6,
        column_3p=7
    )

    # Add a small constant to avoid log(0)
    small_constant = 1e-10
    matrix_safe = combined_matrix + small_constant

    # Apply the natural logarithm
    log_matrix = np.log(matrix_safe)
    combined_matrix = log_matrix

    # Iterate over k_iter for GMM clustering
    for k_iter in range(2, 5):
        print(f'Running clustering for k_iter = {k_iter}...')
        
        # Perform GMM clustering
        gmm_model, gmm_cluster_assignments, gmm_probabilities = perform_gmm(
            combined_matrix, n_components=k_iter)
        
        gmm_plot_path = create_plot_directories(plot_path, "GMM", k_iter)
        
        plot_cluster_probabilities_sorted_multicol(
            gmm_probabilities, sample_names, gmm_plot_path, k_iter,
            pdf_filename=f"cluster_report_k{k_iter}.pdf")
        pdf1 = os.path.join(gmm_plot_path, f"cluster_report_k{k_iter}.pdf")
        
        plot_weighted_profiles(
            gmm_probabilities, sample_names, input_dir, gmm_plot_path, True,
            "GMM", k_iter, pdf_filename=f"weighted_profiles_k{k_iter}.pdf")
        pdf2 = os.path.join(gmm_plot_path, f"weighted_profiles_k{k_iter}.pdf")
    
        # Make PCA Plot with gradient
        transformed_data, explained_variance, pca = perform_pca(combined_matrix, None)
        pca_plot_path = create_plot_directories(plot_path, "PCA", k_iter)
        plot_pca_gradient(
            transformed_data, sample_names, gmm_probabilities, pca_plot_path,
            explained_variance=explained_variance, pdf_filename=f"pca_k{k_iter}.pdf")
        pdf3 = os.path.join(pca_plot_path, f"pca_k{k_iter}.pdf")
    
        pdf_list = [pdf1, pdf2, pdf3]
        
        # Merge the PDFs into a single file
        merged_pdf_path = os.path.join(plot_path, f"damage_report_k{k_iter}.pdf")
        merge_pdfs(pdf_list, merged_pdf_path)
    
        # Scale the final merged PDF to match the width of the cluster report
        target_width = 8.5 * 72  # 72 points per inch
        scaled_pdf_path = os.path.join(plot_path, f"damage_report_k{k_iter}.pdf")
        scale_pdf(merged_pdf_path, scaled_pdf_path, target_width)
    
        print(f"PDFs merged and scaled to {target_width/72:.2f} inches width. Final PDF: {scaled_pdf_path}")

if __name__ == '__main__':
    main()
