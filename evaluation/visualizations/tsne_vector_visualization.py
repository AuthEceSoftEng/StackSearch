#!/usr/bin/env python

import os
import json
import pickle
import numpy as np
import pandas as pd
from spacy.lang import en

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from bokeh.palettes import brewer
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import HoverTool, WheelZoomTool, PanTool, BoxZoomTool, ResetTool, value

def get_pca_vectors(input_vec, n_components=200):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(input_vec)

def get_tsne_vectors(input_vec, perplexity=40, early_exaggeration=10, 
                n_iter=2500, v=2, serialize=False, export_dir=None, name=''):
    tsne = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration, 
        n_iter=n_iter, verbose=v)
    tsne_vectors = tsne.fit_transform(input_vec)
    # save t-SNE model and vectors to disk
    if serialize:
        tsne_model_path = os.path.join(export_dir, 'tsne_' + name + '_model.pkl')
        with open(tsne_model_path, 'wb') as out:
            pickle.dump(tsne, out)
        tsne_vecs_path = os.path.join(export_dir, 'tsne_' + name + '_vectors.npy')
        np.save(tsne_vecs_path, tsne_vectors)
        print('t-SNE model and vectors saved at', os.path.realpath(export_dir))
    return tsne_vectors

def filter_unspecified(bvecs, tvecs, post_tags):
    new_bvecs = None
    new_tvecs = None
    new_post_tags = []
    for idx, tag in enumerate(post_tags):
        if tag != 'unspecified':
            new_post_tags.append(tag)
            if new_bvecs is None:
                new_bvecs = bvecs[idx]
                new_tvecs = tvecs[idx]
            else:
                new_bvecs = np.vstack((new_bvecs, bvecs[idx]))
                new_tvecs = np.vstack((new_tvecs, tvecs[idx]))
    return new_bvecs, new_tvecs, new_post_tags

def sort_by_score(data, metadata_path):
    with open(metadata_path, 'rb') as _in:
        metadata = json.load(_in)

    scores = []
    for post in metadata:
        scores.append(post['Score'])
    sorted_idx = np.argsort(scores)

    sorted_data = None
    if isinstance(data, np.ndarray):
        sorted_data = np.zeros(data.shape, dtype=data.dtype)
        for ii, idx in enumerate(sorted_idx[::-1]):
            sorted_data[ii] = data[idx]
    elif isinstance(data, list):
        sorted_data = []
        for idx in sorted_idx[::-1]:
            sorted_data.append(data[idx])

    return sorted_data

def assign_colors(post_tags, highlight_tags):
    palette = brewer['Set1'][len(highlight_tags)]
    colors = []
    tag_names = []
    for tag in post_tags:
        if tag in highlight_tags:
            tag_names.append(tag)
            colors.append(palette[highlight_tags.index(tag)])
        else:
            tag_names.append(tag)
            colors.append('#d3d3d3')
    return colors, tag_names

def build_legend(post_tags, highlight_tags):
    legend = []
    for tag in post_tags:
        if tag in highlight_tags:
            legend.append(tag)
        else:
            legend.append('unspecified')
    return legend

def posts_with_specific_tags(metadata_path, highlight_tags):
    with open(metadata_path, 'rb') as _in:
        metadata = json.load(_in)
    post_tags = []
    for post in metadata:
        no_tag = True
        for tag in post['Tags']:
            if tag in highlight_tags and len(post['Tags']) <= 5:
                no_tag = False
                post_tags.append(tag)
                break
        if no_tag:
            #post_tags.append('unspecified')
            post_tags.append(', '.join(post['Tags']))
    return post_tags

def bokeh_plot(dataframe, export_dir, name, plot_size=850):
    png_path = os.path.join(export_dir, 't-SNE_' + name + '_vectors.png')
    html_path = os.path.join(export_dir, 't-SNE_' + name + '_vectors.html')
    title = '[t-SNE] Java/StackOverflow ' + name + ' vectors in 2D space'
    output_file(html_path, title=title)
    # plot tools
    hover = HoverTool(tooltips = '@tag')
    tools = [hover, WheelZoomTool(), BoxZoomTool(), ResetTool(), PanTool()]
    #tools = [WheelZoomTool(), BoxZoomTool(), ResetTool(), PanTool()]
    # add DataFrame as a ColumnDataSource for Bokeh
    plot_data = ColumnDataSource(dataframe)
    # create the plot and configure the title, dimensions, and tools
    tsne_plot = figure(
                        title=title,
                        tools=tools,
                        plot_width = plot_size,
                        plot_height = plot_size)
    tsne_plot.circle(
                    'x_coord', 'y_coord',
                    source=plot_data, color='color', legend='legend',
                    line_alpha=0.2, fill_alpha=0.5, size=10, hover_line_color='black')
    # configure visual elements of the plot
    tsne_plot.title.text_font_size = value('16pt')
    tsne_plot.xaxis.visible = False
    tsne_plot.yaxis.visible = False
    tsne_plot.grid.grid_line_color = None
    tsne_plot.outline_line_color = None
    # engage!
    show(tsne_plot)

def main(vector_index_path, metadata_path, highlight_tags, export_dir, 
                                        sample_size=4000, pca_smooth=True):
    with open(vector_index_path, 'rb') as _in:
        vec_index = pickle.load(_in)
    bvecs = vec_index['BodyVectors']
    tvecs = vec_index['TitleVectors']

    if pca_smooth:
        bvecs = get_pca_vectors(bvecs)
        tvecs = get_pca_vectors(tvecs)

    post_tags = posts_with_specific_tags(metadata_path, highlight_tags)
  
    # sort by score and get top sample_size rows
    bvecs = sort_by_score(bvecs, metadata_path)[:sample_size]
    tvecs = sort_by_score(tvecs, metadata_path)[:sample_size]
    post_tags = sort_by_score(post_tags, metadata_path)[:sample_size]
    #bvecs, tvecs, post_tags = filter_unspecified(bvecs, tvecs, post_tags)
    colors, tag_names = assign_colors(post_tags, highlight_tags)
    legend = build_legend(post_tags, highlight_tags)

    
    tsne_bvecs = get_tsne_vectors(bvecs, serialize=True, export_dir=export_dir, 
        name='body')
    tsne_tvecs = get_tsne_vectors(tvecs, serialize=True, export_dir=export_dir, 
        name='title')
    
    '''
    tsne_vecs_path = os.path.join(export_dir, 'tsne_body_vectors.npy')
    tsne_bvecs = np.load(tsne_vecs_path)
    tsne_vecs_path = os.path.join(export_dir, 'tsne_title_vectors.npy')
    tsne_tvecs = np.load(tsne_vecs_path)
    '''

    bdframe = pd.DataFrame(tsne_bvecs, columns=['x_coord', 'y_coord'])
    bdframe['color'] = colors
    bdframe['tag'] = tag_names
    bdframe['legend'] = legend
    bokeh_plot(bdframe, export_dir, 'body')
    
    tdframe = pd.DataFrame(tsne_tvecs, columns=['x_coord', 'y_coord'])
    tdframe['color'] = colors
    tdframe['tag'] = tag_names
    tdframe['legend'] = legend
    bokeh_plot(tdframe, export_dir, 'title')
    
    
if __name__ == '__main__':
    demo_tags = ['junit', 'jpa', 'jackson', 'multithreading', 'date']
    main('../../src/wordvec_model/index/ft_v0.1.2_post_index.pkl', 
            '../../src/wordvec_model/index/metadata.json', demo_tags, '.')
