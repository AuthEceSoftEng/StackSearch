#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import plotting_context
import matplotlib.pyplot as plt

def load_label_sets(filepath, num_sets):
    with open(filepath, 'r') as f:
        label_list = []
        for row in f:
            label_list.append(int(row))
    label_sets_array = np.array_split(np.array(label_list), num_sets)
    return label_sets_array
    
def avg_precision(labeled_results, k=10):
    """
    Returns The average precision at k.
    """
    labeled_results = labeled_results[:k]
    result_count = 0
    _sum = 0

    for idx, val in enumerate(labeled_results):
        if val == 1:
            result_count += 1
            _sum += result_count/(idx + 1)
    score = _sum / k
    return score

def mean_avg_precision(avgprec_vals, k=10):
    """
    Computes the mean average precision at k.
    """
    return np.mean(avgprec_vals)

def mean_search_length(labeled_result_sets, relevant_results_required=5, max_results=20):
    def search_length(labeled_results, relevant_results_required, max_results):
        relevant_result_count = 0
        irrelevant_result_count = 0
        for label in labeled_results:
            if label == 1:
                relevant_result_count += 1
                if relevant_result_count == relevant_results_required:
                    return irrelevant_result_count
            else:
                irrelevant_result_count += 1
        return max_results
    
    def search_length_list(labeled_result_sets, relevant_results_required, max_results):
        resulting_search_lengths = np.zeros(len(labeled_result_sets), dtype='float32')
        for idx, query_label_set in enumerate(labeled_result_sets):
            resulting_search_lengths[idx] = search_length(query_label_set, relevant_results_required, max_results)
        return resulting_search_lengths
    

    if isinstance(relevant_results_required, list):
        msl = np.zeros(len(relevant_results_required), dtype='float32')
        for idx, num_res in enumerate(relevant_results_required):
            msl[idx] = np.mean(search_length_list(labeled_result_sets, num_res, max_results))
        return msl
    else:
        return np.mean(search_length_list(labeled_result_sets, num_res, max_results))

def main(ft_label_path, tf_label_path, hy_label_path, num_queries):
    with plotting_context('paper', font_scale=2):
        
        ftsets = load_label_sets(ft_label_path, num_queries)
        tfsets = load_label_sets(tf_label_path, num_queries)
        hysets = load_label_sets(hy_label_path, num_queries)
        
        avg_prec_plots = [5, 10, 15, 20]
        ft_map = np.zeros(len(avg_prec_plots), dtype='float32')
        tf_map = np.zeros(len(avg_prec_plots), dtype='float32')
        hy_map = np.zeros(len(avg_prec_plots), dtype='float32')
        ## Average Precision Plots
        for ii, k in enumerate(avg_prec_plots):
            ft_avgp = np.zeros(num_queries, dtype='float32')
            tf_avgp = np.zeros(num_queries, dtype='float32')
            hy_avgp = np.zeros(num_queries, dtype='float32')
            for idx, vset in enumerate(ftsets):
                ft_avgp[idx] = avg_precision(vset, k)
            for idx, vset in enumerate(tfsets):
                tf_avgp[idx] = avg_precision(vset, k)
            for idx, vset in enumerate(hysets):
                hy_avgp[idx] = avg_precision(vset, k)
            
            ft_map[ii] = np.mean(ft_avgp)
            tf_map[ii] = np.mean(tf_avgp)
            hy_map[ii] = np.mean(hy_avgp)

            system_labels = ['FastText'] * num_queries + ['TF-IDF'] * num_queries + ['Hybrid'] * num_queries
            init_xlabels = ['Q' + str(i + 1) for i in range(num_queries)]
            x_labels = init_xlabels * 3
            y_values = np.hstack((ft_avgp, tf_avgp, hy_avgp))
            plot_df = pd.DataFrame({'Query': x_labels, 'Score': y_values, 'System': system_labels})
            sns.set_style('whitegrid', {'legend.frameon': 'True'})
            g = sns.factorplot(x='Query', y='Score', hue='System', palette='Set1', legend=False, data=plot_df, kind='bar')
            g.despine(left=True, bottom=True)
            g.set(ylim=(0.0, 1.0))
            g.fig.suptitle('Average Precision at {} Results'.format(k))
            plt.legend(loc='upper right')
            plt.xlim(-1,18)
            plt.subplots_adjust(right=0.96, left=0.06, top=0.9)
        
        ## Mean Average Precision Plot
        system_labels = ['FastText'] * len(ft_map) + ['TF-IDF'] * len(tf_map) + ['Hybrid'] * len(hy_map)
        y_values = np.hstack((ft_map, tf_map, hy_map))
        x_labels = ['K = 5', 'K = 10', 'K = 15', 'K = 20'] * 3
        plot_df = pd.DataFrame({'Number of Results (K)': x_labels, 'Score': y_values, 'System': system_labels})
        sns.set_style('whitegrid', {'legend.frameon': 'True'})
        map_plt = sns.factorplot(x='Number of Results (K)', y='Score', hue='System', palette='Set1', legend=False, data=plot_df, kind='bar')
        map_plt.despine(left=True, bottom=True)
        map_plt.fig.suptitle('Mean Average Precision (MAP)')
        map_plt.fig.subplots_adjust(top=.9)
        plt.legend(loc='upper right')
        plt.xlim(-1, len(ft_map))
        ax = plt.gca()
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
                fontsize=12, color='black', ha='center', va='bottom')
        plt.subplots_adjust(right=0.96, left=0.06, top=0.9)

        ## Mean Search Length Plot
        ft_msl = mean_search_length(ftsets, [5, 10])
        tf_msl = mean_search_length(tfsets, [5, 10])
        hy_msl = mean_search_length(hysets, [5, 10])
        system_labels = ['FastText'] * len(ft_msl) + ['TF-IDF'] * len(tf_msl) + ['Hybrid'] * len(hy_msl)
        y_values = np.hstack((ft_msl, tf_msl, hy_msl))
        x_labels = ['n = 5', 'n = 10'] * 3
        plot_df = pd.DataFrame({'Number of Relevant Results Required (n)': x_labels, 'Score': y_values, 'System': system_labels})
        sns.set_style('whitegrid', {'legend.frameon': 'True'})
        msl_plt = sns.factorplot(x='Number of Relevant Results Required (n)', y='Score', hue='System', palette='Set1', legend=False, data=plot_df, kind='bar')
        msl_plt.despine(left=True, bottom=True)
        msl_plt.fig.suptitle('Mean Search Length (MSL)')
        plt.legend(loc='upper right')
        plt.xlim(-1, len(ft_msl))
        ax = plt.gca()

        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.2f}'.format(p.get_height()), 
                fontsize=12, color='black', ha='center', va='bottom')
        plt.subplots_adjust(right=0.96, left=0.06, top=0.9)
        
        ## Postlink Eval plot
        ft_res = np.array([19.95, 14.98, 21.34])
        tf_res = np.array([22.14, 18.98, 25.47])
        hy_res = np.array([24.2, 20.21, 27.87])
        system_labels = ['FastText'] * len(ft_res) + ['TF-IDF'] * len(tf_res) + ['Hybrid'] * len(hy_res)
        y_values = np.hstack((ft_res, tf_res, hy_res))
        x_labels = ['Title', 'Body', 'Title-Body'] * 3
        plot_df = pd.DataFrame({'Search Mechanism Setting': x_labels, '% of PostLinks found': y_values, 'System': system_labels})
        sns.set_style('whitegrid', {'legend.frameon': 'True'})
        msl_plt = sns.factorplot(x='Search Mechanism Setting', y='% of PostLinks found', hue='System', palette='Set1', legend=False, data=plot_df, kind='bar')
        msl_plt.despine(left=True, bottom=True)
        msl_plt.fig.suptitle('PostLink Evaluation')
        plt.legend(loc='upper right')
        plt.xlim(-1, len(ft_res))
        ax = plt.gca()

        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.2f}%'.format(p.get_height()), 
                fontsize=12, color='black', ha='center', va='bottom')
        plt.subplots_adjust(right=0.96, left=0.06, top=0.9)
        plt.show()


if __name__ == '__main__':
    main('results_fasttext', 'results_tfidf', 'results_hybrid', 18)
