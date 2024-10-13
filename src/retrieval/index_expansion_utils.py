from nltk import sent_tokenize


def fetch_expansion_from_cache(index_expansion_result_cache, cur_sess_id):
    processed_id = cur_sess_id.replace('answer_', '').replace('noans_', '')
    # candidate_results = [v for k, v in index_expansion_result_cache.items() if processed_id in k]
    try:
        cur_expansion = index_expansion_result_cache[processed_id]
    except:
        # failure to generate the expansion; index_expansion_result_cache could also contain None values.
        cur_expansion = None
    if type(cur_expansion) == str:
        cur_expansion = [cur_expansion]
    return cur_expansion
    

def resolve_expansion(expansion_type, resolution_strategy, 
                      existing_corpus, existing_corpus_ids, existing_corpus_timestamps,
                      cur_item_expansions, cur_sess_id, ts):
    # preprocess and split expansion, if applicable
    if expansion_type == 'session-summ':
        # print(cur_item_expansions)
        if cur_item_expansions is None:
            cur_item_expansions = ['']
        assert len(cur_item_expansions) == 1
        if 'split' in resolution_strategy:
            cur_item_expansions = sent_tokenize(cur_item_expansions[0])
    elif expansion_type == 'session-keyphrase' or expansion_type == 'turn-keyphrase':
        if cur_item_expansions is None:
            cur_item_expansions = ['']
            # print('Warning: none value for keyphrase expansion')
        assert len(cur_item_expansions) == 1
        if 'split' in resolution_strategy:
            cur_item_expansions = [x.strip() for x in cur_item_expansions[0].split(';')]
    elif expansion_type == 'session-userfact' or expansion_type == 'turn-userfact':
        if cur_item_expansions is None:
            # For failed expansions, we treat it as if the expansion is an empty string
            cur_item_expansions = [""]
        cur_item_expansions = [str(x) for x in cur_item_expansions]
        if 'split' not in resolution_strategy:
            if cur_item_expansions:
                cur_item_expansions = [' '.join(cur_item_expansions)]
            else:
                cur_item_expansions = []
    else:
        raise NotImplementedError

    # merge expansion with the main items
    if 'separate' in resolution_strategy:
        existing_corpus += [str(x) for x in cur_item_expansions]
        existing_corpus_ids += [cur_sess_id for _ in cur_item_expansions]
        existing_corpus_timestamps += [ts for _ in cur_item_expansions]
    elif 'merge' in resolution_strategy or 'replace' in resolution_strategy:
        out_corpus, out_corpus_ids, out_corpus_timestamps = [], [], []
        N = len(existing_corpus_ids)
        for i in range(N):
            if existing_corpus_ids[i] == cur_sess_id:
                if 'merge' in resolution_strategy:
                    for expansion_item in cur_item_expansions:
                        out_corpus.append(expansion_item + ' ' + existing_corpus[i])
                        #print(existing_corpus[i], '--->', expansion_item + ' ' + existing_corpus[i])
                        #print('\n\n+++\n\n')
                        out_corpus_ids.append(existing_corpus_ids[i])
                        out_corpus_timestamps.append(existing_corpus_timestamps[i])
                elif 'replace' in resolution_strategy:
                    for expansion_item in cur_item_expansions:
                        out_corpus.append(expansion_item)  # different
                        out_corpus_ids.append(existing_corpus_ids[i])
                        out_corpus_timestamps.append(existing_corpus_timestamps[i])
                else:
                    raise NotImplementedError
            else:
                out_corpus.append(existing_corpus[i])
                out_corpus_ids.append(existing_corpus_ids[i])
                out_corpus_timestamps.append(existing_corpus_timestamps[i])
        existing_corpus, existing_corpus_ids, existing_corpus_timestamps = out_corpus, out_corpus_ids, out_corpus_timestamps
    else:
        raise NotImplementedError
    
    return existing_corpus, existing_corpus_ids, existing_corpus_timestamps
    
