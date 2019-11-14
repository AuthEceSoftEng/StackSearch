#!/usr/bin/env python

import os
import glob
import json
import time
import pickle
import sqlite3
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

## Params
min_postlinks_per_id = 2 
min_snippets_per_id = 1

## Errors
shape_error = 'Inconsistent number of rows for TFIDF & FastText matrices.\nTFIDF_shape: {}\nFastText_shape: {}'

## File Paths
postlinks_db = '../../src/database/javapostlinks.db'
index_dataset_path = '../../src/wordvec_models/index/data/index_dataset.pkl'

## Export Paths
alg0_dir = 'alg_zero_results'
index_sets_dir = 'index_sets'
temp_dir = 'temp_files'

# Index
ft_version = 'v0.6.1'
fasttext_index_path = '../../src/wordvec_models/index/ft_' + ft_version + '_post_index.pkl'
tfidf_version = 'v0.3'
tfidf_index_path = '../../src/wordvec_models/index/tfidf_' + tfidf_version + '_post_index.pkl'
glove_version = 'v0.1.1'
glove_index_path = '../../src/wordvec_models/index/glove_' + glove_version + '_post_index.pkl'

# Metadata
metadata_path = '../../src/wordvec_models/index/metadata.json'

## PostLink Query
linked_posts_query = '''SELECT DISTINCT PostId, RelatedPostId FROM postlinks 
WHERE PostId IN {} AND RelatedPostId IN {}'''


def build_postlink_lookup(post_ids, export_dir, min_id_links=3):
    postlink_lookup = []
    postid_freqs = OrderedDict()
    postids_str = str(tuple(post_ids))
    pldb = sqlite3.connect(postlinks_db)
    c = pldb.cursor()
    c.execute(linked_posts_query.format(postids_str, postids_str))
    for row in c:
        if row[0] not in postid_freqs:
            postid_freqs[row[0]] = 0
        postid_freqs[row[0]] += 1
        if row[1] not in postid_freqs:
            postid_freqs[row[1]] = 0
        postid_freqs[row[1]] += 1
    finalpostids = []
    for key, value in postid_freqs.items():
        if value >= min_id_links:
            finalpostids.append(key)
    print('{} postids with appearance frequency >= {}'.format(
        len(finalpostids), min_id_links))
    postids_str = str(tuple(finalpostids))
    c.execute(linked_posts_query.format(postids_str, postids_str))
    for row in c:
        linked_str = str(row[0]) + '-' + str(row[1])
        rev_linked_str = str(row[1]) + '-' + str(row[0])
        postlink_lookup.append(linked_str)
        postlink_lookup.append(rev_linked_str)
    fs_postlink_lookup = frozenset(postlink_lookup)
    with open(os.path.join(export_dir, 'postlink_lookup.pkl'), 'wb') as out:
        pickle.dump(fs_postlink_lookup, out)
    print('lookup saved at', os.path.join(export_dir, 'postlink_lookup.pkl'))
    return fs_postlink_lookup


def build_postlink_searches(index_sets, export_dir):
    searches = []
    for iset in index_sets:
        for pindex in iset[1:]:
            searches.append(str(iset[0]) + '-' + str(pindex))
    with open(os.path.join(export_dir, 'searches.pkl'), 'wb') as out:
        pickle.dump(searches, out)
    print('searches saved at', os.path.join(export_dir, 'searches.pkl'))
    return searches


def postlink_percentage_alg_1(searches, lookup):
    count = 0
    slen = len(searches)
    div = int(len(lookup) / 2)
    for idx, s in enumerate(searches):
        print('\r{}/{} - {}/{}'.format((idx + 1), slen, count, div), end='')
        if s in lookup:
            count += 1
    pavg = (count / div)
    postlink_per = round(100 * pavg, 2)
    print('\nHit percentage:', postlink_per)
    return postlink_per


def postlink_percentage_alg_2(title_vecs, body_vecs, t=0.85):
    """Uses a threshold value for similar posts for a more granular
    approach.
    """
    def retrieve_postid_pairs(post_ids):
        postid_pairs = []
        postids_str = str(tuple(post_ids))
        pldb = sqlite3.connect(postlinks_db)
        c = pldb.cursor()
        c.execute(linked_posts_query.format(postids_str, postids_str))
        for row in c:
            postid_pairs.append(row)
        return postid_pairs

    def postid_pairs_to_index_pairs(postid_pairs, index):
        index_pairs = []
        for pair in postid_pairs:
            ipair = (index.get_loc(pair[0]), index.get_loc(pair[1]))
            index_pairs.append(ipair)
        return index_pairs

    index = pd.read_pickle(index_dataset_path).index
    pid_pairs = retrieve_postid_pairs(list(index))
    ipairs = postid_pairs_to_index_pairs(pid_pairs, index)
    ipairs_len = len(ipairs)
    print(ipairs_len, 'postlink pairs')
    pl_found = {'title': 0, 'body': 0, 'title-body': 0}

    for ii, p in enumerate(ipairs):
        print('\r', ii + 1, '/', ipairs_len, end='')
        title_sim = cosine_similarity(title_vecs[p[0]].reshape(1, -1),
                                      title_vecs[p[1]].reshape(1, -1))
        body_sim = cosine_similarity(body_vecs[p[0]].reshape(1, -1),
                                     body_vecs[p[1]].reshape(1, -1))
        comb_sim = (title_sim + body_sim) / 2
        if title_sim >= t:
            pl_found['title'] += 1
        if body_sim >= t:
            pl_found['body'] += 1
        if comb_sim >= t:
            pl_found['title-body'] += 1

    print()
    print(pl_found)
    perc_found = {
        key: round(100 * (pl_found[key] / ipairs_len), 2)
        for key in pl_found.keys()
    }
    print(perc_found)


def fetch_postids_from_metadata(metadata_path, index_sets, min_snippets=1):
    print('Fetching PostId sets from metadata.')
    id_sets = []
    with open(metadata_path, 'rb') as _in:
        metadata = json.load(_in)
    for idx, iset in enumerate(index_sets):
        if metadata[idx]['SnippetCount'] >= min_snippets:
            id_sets.append([metadata[idx]['PostId']] +
                           [metadata[i]['PostId'] for i in iset])
    return id_sets


def matrix_batches(matrix, batch_size):
    """Simple function implementation for matrix batching"""
    indices = list(range(0, matrix.shape[0] + batch_size, batch_size))
    batched_matrix = [matrix[i:(i + batch_size)] for i in indices]
    # Remove empty extra batches from the end
    while (batched_matrix[-1].shape[0] == 0):
        del batched_matrix[-1]
    return batched_matrix


def load_index_sets(export_dir, prefix, num_results=20, partitioned=True):
    index_sets = None
    file_list = glob.glob(os.path.join(export_dir, '*.npy'))
    file_list = [
        filename for filename in file_list
        if os.path.basename(filename).startswith(prefix)
    ]
    print(file_list)
    if partitioned:
        index_sets = np.load(file_list[0]).astype(np.int)
        # [[..], ..., [..]] combined sims
        print(index_sets.shape)
        if len(index_sets.shape) == 2:
            for _file in file_list[1:]:
                index_sets = np.concatenate(
                    (index_sets, np.load(_file).astype(np.int)), axis=0)
        # [[[..], ..., [..]], [[..], ..., [..]], [[..], ..., [..]]] separate & combined sims
        elif len(index_sets.shape) == 3:
            for _file in file_list[1:]:
                index_sets = np.concatenate(
                    (index_sets, np.load(_file).astype(np.int)), axis=1)
        else:
            raise ValueError('Expected 2-D or 3-D matrix, not {}.'.format(
                len(index_sets.shape)))
        print('index_sets shape:', index_sets.shape)
        np.save(
            os.path.join(export_dir, prefix + '_sorted_index_sets.npy'),
            index_sets)
        print('sorted index sets saved at:',
              os.path.join(export_dir, 'sorted_index_sets.npy'))
    else:
        index_sets = np.load(file_list[0])
    if num_results < 20:
        if len(index_sets.shape) == 2:
            index_sets = index_sets[:, :num_results]
        elif len(index_sets.shape) == 3:
            index_sets = index_sets[:, :, :num_results]
        else:
            raise ValueError('Expected 2-D or 3-D matrix, not {}.'.format(
                len(index_sets.shape)))
    return index_sets


def calc_hybrid_batch_sims_rerank(tfidf_tvecs,
                                  tfidf_bvecs,
                                  ft_tvecs,
                                  ft_bvecs,
                                  rerank_t=200,
                                  num_set_ids=20,
                                  batch_size=60,
                                  version='v0'):

    if tfidf_tvecs.shape[0] != ft_tvecs.shape[0]:
        print(tfidf_tvecs.shape, ft_tvecs.shape)
        raise ValueError(
            shape_error.format(tfidf_tvecs.shape[0], ft_tvecs.shape[0]))
    index_sets = np.zeros([3, len(ft_tvecs), num_set_ids], dtype='int32')
    batched_tfidf_qvecs = matrix_batches(tfidf_tvecs, batch_size)
    batched_ft_qvecs = matrix_batches(ft_tvecs, batch_size)
    n_batches = len(batched_ft_qvecs)
    if len(batched_ft_qvecs) != len(batched_tfidf_qvecs):
        raise ValueError('Inconsistent batched matrices.')
    print('Data length:', len(ft_tvecs))
    print(n_batches, 'batches')
    istart = 0
    for idx, tfidf_batch in enumerate(batched_tfidf_qvecs):
        # Sanity check
        batch_size = tfidf_batch.shape[0]
        if batch_size > 0:
            start_time = time.time()
            # query - title & query - body similarities
            ## TFIDF similarities
            tsims_batch = -cosine_similarity(tfidf_batch, tfidf_tvecs)
            bsims_batch = -cosine_similarity(tfidf_batch, tfidf_bvecs)
            # 0.5 weight for both title and body similarities (combined similarity)
            csims_batch = (tsims_batch + bsims_batch) / 2
            # Top rerank_t sorted indices from TFIDF
            tsi = np.argsort(tsims_batch)[:, :rerank_t]
            bsi = np.argsort(bsims_batch)[:, :rerank_t]
            csi = np.argsort(csims_batch)[:, :rerank_t]
            # Creating new batches for fastText
            vdim = ft_tvecs.shape[1]
            tft_batch = np.zeros((batch_size, rerank_t, vdim), dtype=np.int32)
            bft_batch = np.zeros((batch_size, rerank_t, vdim), dtype=np.int32)
            cft_batch = np.zeros(
                (2, batch_size, rerank_t, vdim), dtype=np.int32)
            for ii in range(batch_size):
                tft_batch[ii] = np.take(ft_tvecs, tsi[ii], axis=0)
                bft_batch[ii] = np.take(ft_bvecs, bsi[ii], axis=0)
                cft_batch[0][ii] = np.take(ft_tvecs, csi[ii], axis=0)
                cft_batch[1][ii] = np.take(ft_bvecs, csi[ii], axis=0)
            # fastText re-ranking
            tftsims_batch, bftsims_batch, cftsims_batch = [
                np.zeros((batch_size, rerank_t)) for i in range(3)
            ]
            for ii in range(batch_size):
                bftvecs = batched_ft_qvecs[idx][ii].reshape(1, -1)
                tftsims_batch[ii] = (
                    -cosine_similarity(bftvecs, tft_batch[ii])).reshape(-1)
                bftsims_batch[ii] = (
                    -cosine_similarity(bftvecs, bft_batch[ii])).reshape(-1)
                cftsims_batch[ii] = (
                    (-cosine_similarity(bftvecs, cft_batch[0][ii]) -
                     cosine_similarity(bftvecs, cft_batch[1][ii])) /
                    2).reshape(-1)
            # Sort top rerank_t indices based on new similarities
            new_tsi, new_bsi, new_csi = [
                np.zeros((batch_size, rerank_t), dtype=np.int32)
                for i in range(3)
            ]
            for ii in range(batch_size):
                new_tsi[ii] = tsi[ii][np.argsort(tftsims_batch[ii])]
                new_bsi[ii] = bsi[ii][np.argsort(bftsims_batch[ii])]
                new_csi[ii] = csi[ii][np.argsort(cftsims_batch[ii])]
            # Retain top num_set_ids sorted indices
            index_sets[0, istart:(
                istart + batch_size)] = new_tsi[:, :num_set_ids]
            index_sets[1, istart:(
                istart + batch_size)] = new_bsi[:, :num_set_ids]
            index_sets[2, istart:(
                istart + batch_size)] = new_csi[:, :num_set_ids]
            istart = istart + batch_size
            end_time = time.time()
            rem_time = (end_time - start_time) * (n_batches - (idx + 1))
            rem_time_str = str(datetime.timedelta(seconds=rem_time))
            print('\rbatch #', idx + 1, 'rem_time:', rem_time_str, end='')
        else:
            break
    print()
    path = os.path.join(index_sets_dir,
                        '_'.join(['hybrid', 'rerank', version, '20res']))
    np.save(path, index_sets)
    return index_sets


def calc_hybrid_batch_sims(tfidf_tvecs,
                           tfidf_bvecs,
                           ft_tvecs,
                           ft_bvecs,
                           num_set_ids=20,
                           batch_size=60,
                           version='v0'):

    if tfidf_tvecs.shape[0] != ft_tvecs.shape[0]:
        print(tfidf_tvecs.shape, ft_tvecs.shape)
        raise ValueError(
            shape_error.format(tfidf_tvecs.shape[0], ft_tvecs.shape[0]))
    index_sets = np.zeros([3, len(ft_tvecs), num_set_ids], dtype='int32')
    batched_tfidf_qvecs = matrix_batches(tfidf_tvecs, batch_size)
    batched_ft_qvecs = matrix_batches(ft_tvecs, batch_size)
    n_batches = len(batched_ft_qvecs)
    if len(batched_ft_qvecs) != len(batched_tfidf_qvecs):
        raise ValueError('Inconsistent batched matrices.')
    print('Data length:', len(ft_tvecs))
    print(n_batches, 'batches')
    istart = 0
    for idx, tfidf_batch in enumerate(batched_tfidf_qvecs):
        # Sanity check
        if tfidf_batch.shape[0] > 0:
            start_time = time.time()
            # query - title & query - body similarities
            ## TFIDF & FastText hybrid similarities
            tsims_batch = (
                -cosine_similarity(tfidf_batch, tfidf_tvecs) -
                cosine_similarity(batched_ft_qvecs[idx], ft_tvecs)) / 2
            bsims_batch = (
                -cosine_similarity(tfidf_batch, tfidf_bvecs) -
                cosine_similarity(batched_ft_qvecs[idx], ft_bvecs)) / 2
            # 0.5 weight for both title and body similarities (combined similarity)
            csims_batch = (tsims_batch + bsims_batch) / 2
            # Retain top num_set_ids sorted indices
            index_sets[0, istart:(istart + tfidf_batch.shape[0]
                                  )] = np.argsort(tsims_batch)[:, :num_set_ids]
            index_sets[1, istart:(istart + tfidf_batch.shape[0]
                                  )] = np.argsort(bsims_batch)[:, :num_set_ids]
            index_sets[2, istart:(istart + tfidf_batch.shape[0]
                                  )] = np.argsort(csims_batch)[:, :num_set_ids]
            istart = istart + tfidf_batch.shape[0]
            end_time = time.time()
            rem_time = (end_time - start_time) * (n_batches - (idx + 1))
            rem_time_str = str(datetime.timedelta(seconds=rem_time))
            print('\rbatch #', idx + 1, 'rem_time:', rem_time_str, end='')
        else:
            break
    print()
    path = os.path.join(index_sets_dir, '_'.join(['hybrid', version, '20res']))
    np.save(path, index_sets)
    return index_sets


def calc_batch_sims(title_vecs,
                    body_vecs,
                    start_index,
                    part_size,
                    num_set_ids=20,
                    batch_size=120,
                    model='ft',
                    version='v0'):
    """Treat all titles as queries 
    Get cosine distances of every query and every title
    Do the same for queries (titles) and bodies
    """

    print('Calculating similarity matrices and returning sorted index sets.')
    if part_size + start_index > title_vecs.shape[0]:
        end_index = title_vecs.shape[0]
        part_size = end_index - start_index
    else:
        end_index = start_index + part_size
    index_sets = np.zeros([3, part_size, num_set_ids], dtype='int32')
    n_batches = round(part_size / batch_size)
    batched_query_vecs = None
    if model == 'tfidf':
        batched_query_vecs = matrix_batches(title_vecs, batch_size)
    else:
        batched_query_vecs = np.array_split(title_vecs[start_index:end_index],
                                            n_batches)
    print('Partition:', start_index, '-', end_index)
    print(len(batched_query_vecs), 'batches')
    istart = 0
    for idx, batch in enumerate(batched_query_vecs):
        if batch.shape[0] > 0:
            start_time = time.time()
            # query - title & query - body similarities
            title_sims_batch = -cosine_similarity(batch, title_vecs)
            body_sims_batch = -cosine_similarity(batch, body_vecs)
            # 0.5 weight for both title and body similarities (combined similarity)
            comb_sims_batch = (title_sims_batch + body_sims_batch) / 2
            # Retain top num_set_ids sorted indices
            index_sets[0, istart:(istart + batch.shape[0])] = np.argsort(
                title_sims_batch)[:, :num_set_ids]
            index_sets[1, istart:(istart + batch.shape[0])] = np.argsort(
                body_sims_batch)[:, :num_set_ids]
            index_sets[2, istart:(istart + batch.shape[0])] = np.argsort(
                comb_sims_batch)[:, :num_set_ids]
            istart = istart + batch.shape[0]
            end_time = time.time()
            rem_time = (end_time - start_time) * (n_batches - (idx + 1))
            rem_time_str = str(datetime.timedelta(seconds=rem_time))
            print('\rbatch #', idx + 1, 'rem_time:', rem_time_str, end='')
        else:
            break
    print()
    path = os.path.join(index_sets_dir, '_'.join([model, version, '20res']))
    np.save(path, index_sets)
    return index_sets


def calc_batch_sims_alg0(title_vecs,
                         body_vecs,
                         t,
                         start_index,
                         part_size,
                         batch_size=120,
                         model='ft',
                         version='v0'):
    """Treat all titles as queries 
    Get cosine distances of every query and every title
    Do the same for queries (titles) and bodies
    """

    indices_path = os.path.join(alg0_dir, '_'.join([model, version, 'index']))
    sims_path = os.path.join(alg0_dir, '_'.join([model, version, 'sims']))
    print('Calculating similarity matrices and returning sorted index sets.')
    if part_size + start_index > title_vecs.shape[0]:
        end_index = title_vecs.shape[0]
        part_size = end_index - start_index
    else:
        end_index = start_index + part_size

    n_batches = round(part_size / batch_size)
    batched_query_vecs = None
    if model == 'tfidf':
        batched_query_vecs = matrix_batches(title_vecs, batch_size)
    else:
        batched_query_vecs = np.array_split(title_vecs[start_index:end_index],
                                            n_batches)
    print('Partition:', start_index, '-', end_index)
    print(len(batched_query_vecs), 'batches')
    istart = 0
    with open(indices_path, 'a') as iout:
        with open(sims_path, 'a') as sout:
            for idx, batch in enumerate(batched_query_vecs):
                if batch.shape[0] > 0:
                    start_time = time.time()
                    # query - title & query - body similarities
                    title_sims_batch = -cosine_similarity(batch, title_vecs)
                    body_sims_batch = -cosine_similarity(batch, body_vecs)
                    # 0.5 weight for both title and body similarities (combined similarity)
                    comb_sims_batch = (title_sims_batch + body_sims_batch) / 2
                    # Sort similarities
                    title_ssims = np.sort(title_sims_batch)
                    body_ssims = np.sort(body_sims_batch)
                    comb_ssims = np.sort(comb_sims_batch)
                    # Sort indices
                    title_sindex = np.argsort(title_sims_batch)
                    body_sindex = np.argsort(body_sims_batch)
                    comb_sindex = np.argsort(comb_sims_batch)
                    # Truth matrix for similarity threshold value
                    title_tm = title_ssims < -t
                    body_tm = body_ssims < -t
                    comb_tm = comb_ssims < -t
                    # Compress and write to files
                    for ii, row in enumerate(title_sindex):
                        np.savetxt(
                            iout, np.compress(title_tm[ii], row), newline=' ')
                        iout.write('\n')
                        np.savetxt(
                            iout,
                            np.compress(body_tm[ii], body_sindex[ii]),
                            newline=' ')
                        iout.write('\n')
                        np.savetxt(
                            iout,
                            np.compress(comb_tm[ii], comb_sindex[ii]),
                            newline=' ')
                        iout.write('\n')

                        np.savetxt(
                            sout,
                            np.compress(title_tm[ii], title_ssims[ii]),
                            newline=' ')
                        sout.write('\n')
                        np.savetxt(
                            sout,
                            np.compress(body_tm[ii], body_ssims[ii]),
                            newline=' ')
                        sout.write('\n')
                        np.savetxt(
                            sout,
                            np.compress(comb_tm[ii], comb_ssims[ii]),
                            newline=' ')
                        sout.write('\n')

                    istart = istart + batch.shape[0]
                    end_time = time.time()
                    rem_time = (end_time - start_time) * (n_batches -
                                                          (idx + 1))
                    rem_time_str = str(datetime.timedelta(seconds=rem_time))
                    print(
                        '\rbatch #',
                        idx + 1,
                        'rem_time:',
                        rem_time_str,
                        end='')
                else:
                    break
    print()


def postlink_eval(index_path, metadata_path, export_dir):
    def eval_process(postids, metadata_path, index_set, export_dir,
                     min_snippets, min_id_links):
        id_sets = fetch_postids_from_metadata(metadata_path, index_set,
                                              min_snippets)
        lookup = build_postlink_lookup(postids, temp_dir, min_id_links)
        searches = build_postlink_searches(id_sets, temp_dir)
        return postlink_percentage_alg_1(searches, lookup)

    
    with open(fasttext_index_path, 'rb') as _in:
        ft_index = pickle.load(_in)

    postlink_percentage_alg_2(ft_index['TitleV'], ft_index['BodyV'], t=0.92)
    
    with open(tfidf_index_path, 'rb') as _in:
        tfidf_index = pickle.load(_in)    

    postlink_percentage_alg_2(tfidf_index['TitleV'], tfidf_index['BodyV'], t=0.75)
    
    
    with open(tfidf_index_path, 'rb') as _in:
        tfidf_index = pickle.load(_in)
        #del tfidf_index['TagV']
    
    
    calc_batch_sims(
        tfidf_index['TitleV'],
        tfidf_index['BodyV'],
        start_index=0,
        part_size=500000,
        batch_size=120,
        model='tfidf'
        version=tfidf_version)
    
    
    with open(fasttext_index_path, 'rb') as _in:
        ft_index = pickle.load(_in)
        #del ft_index['TagsV']  
    
    calc_batch_sims(
        ft_index['TitleV'],
        ft_index['BodyV'],
        start_index=0,
        part_size=500000,
        batch_size=140,
        model='ft',
        version=ft_version)
    
    calc_hybrid_batch_sims(
        tfidf_index['TitleV'],
        tfidf_index['BodyV'],
        ft_index['TitleV'],
        ft_index['BodyV'],
        batch_size=60,
        version=ft_version)
    
    '''
    with open(glove_index_path, 'rb') as _in:
        glove_index = pickle.load(_in)
        
    calc_batch_sims(
        glove_index['TitleV'],
        glove_index['BodyV'],
        start_index=0,
        part_size=500000,
        batch_size=128,
        model='glove',
        version=glove_version)
    '''

    with open(metadata_path, 'rb') as f:
        mdata = json.load(f)
        postids = []
        for p in mdata:
            postids.append(p['PostId'])
        print('number of posts in dataset:', len(postids))

    systems = [
        #'hybrid_rerank_' + ft_version
        #'glove_' + glove_version, 'tfidf_' + tfidf_version, 'hybrid_' + glove_version
        'ft_' + ft_version  , 'tfidf_' + tfidf_version, 'hybrid_' + ft_version
    ]
    sims = ['title', 'body', 'title-body']
    results = {key: {} for key in systems}
    for system in systems:
        index_sets = load_index_sets(export_dir, system, partitioned=False)
        if len(index_sets.shape) > 2:
            for idx, index_set in enumerate(index_sets):
                results[system][sims[idx]] = eval_process(
                    postids, metadata_path, index_set, export_dir,
                    min_snippets_per_id, min_postlinks_per_id)
        else:
            results[system][sims[idx]] = eval_process(
                postids, metadata_path, index_sets, export_dir,
                min_snippets_per_id, min_postlinks_per_id)
    print(results)
    with open('eval_results_' + ft_version + '.json', 'w') as out:
        json.dump(results, out, indent=2)


if __name__ == '__main__':
    postlink_eval(index_dataset_path, metadata_path, index_sets_dir)
