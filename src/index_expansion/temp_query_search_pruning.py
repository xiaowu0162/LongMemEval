import sys
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import openai
import backoff
from src.retrieval.eval_utils import evaluate_retrieval
from datetime import datetime, timedelta


def increment_date(date_str, x):
    # Parse the input date string
    date_obj = datetime.strptime(date_str, "%Y/%m/%d")
    
    # Increment the date by x days
    new_date = date_obj + timedelta(days=x)
    
    # Return the new date in YYYY/MM/DD format
    return new_date.strftime("%Y/%m/%d")


@backoff.on_exception(backoff.constant, (openai.RateLimitError), 
                      interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


openai.organization="YOUR_ORGANZATION"
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url=None,
)


model = 'gpt-4o'
# model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

def infer_time_range(query, query_date):
    system_prompt = "You will be given a question from a human user asking about some prvious events, as well as the time the question is asked. Infer a potential time range such that the events happening in this range is likely to help to answer the question (a start date and an end date). Write a json dict two fields: \"start\" and \"end\". Write date in the form YYYY/MM/DD. If the question does not have any temporal referencea, do not attempt to guess a time range. Instead, just say N/A."

    user_prompt = "Question date: {}\nQuestion:\n{}\n\nRelevant Date Range(dict in json format; do not generate anything else):"

    examples = [
        ({'date': '2023/07/01 (Sat) 23:13', 'question': 'What was the date on which I attended the first BBQ event in June?'}, json.dumps({'start': "2023/06/01", "end": "2023/06/30"})),
        ({'date': '2023/04/10 (Mon) 08:05', 'question': 'Where did I attend the religious activity last week?'}, json.dumps({"start": "2023/04/03", "end": "2023/04/09"})),
        ({'date': '2023/04/01 (Sat) 20:22', 'question': 'What did I do with Rachel on the Wednesday two months ago?'}, json.dumps({"start": "2023/01/25", "end": "2023/02/05"})),
        ({'date': '2023/05/30 (Tue) 01:50', 'question': 'Which pair of shoes did I clean last month?'}, json.dumps({'start': "2023/04/01", "end": "2023/04/30"})),
        ({'date': '2023/04/18 (Tue) 02:06', 'question': 'Who did I meet with during the lunch last Tuesday?'}, json.dumps({"start": "2023/04/10", "end": "2023/04/12"})),
        ({'date': '2023/05/27 (Sat) 01:55', 'question': 'How many months ago did I book the Airbnb in San Francisco?'}, 'N/A'),
        ({'date': '2023/09/04 (Mon) 17:07', 'question': 'How long have I been using my Fitbit Charge 3?'}, 'N/A'),
        ({'date': '2023/10/27 (Fri) 13:00', 'question': 'How many bikes do I currently own?'}, 'N/A'),
        ({'date': '2023/12/18 (Mon) 04:17', 'question': 'What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?'}, 'N/A'),
        ({'date': '2023/11/10 (Fri) 04:20', 'question': 'How many engineers do I lead when I just started my new role as Senior Software Engineer? How many engineers do I lead now?'}, 'N/A'),        
    ]
    
    messages = [{"role": "system", "content": system_prompt}]
    for example_input, example_output in examples:
        messages += [
            {"role": "user", "content": user_prompt.format(example_input['date'], example_input['question'])},
            {"role": "assistant", "content": example_output}
        ]
    messages += [{"role": "user", "content": user_prompt.format(query_date, query)}]

    kwargs = {
        # 'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'model': model, 
        'messages': messages,
        'n': 1,
        'temperature': 0,
        'max_tokens': 2000
    }
    completion = chat_completions_with_backoff(client,**kwargs)
    try:
        out_string = completion.choices[0].message.content.strip()
        out_string = out_string.replace('```json', '')
        out_string = out_string.replace('```', '').strip()
        out_data = json.loads(out_string.strip())
        # hack
        if out_data['start'] in query_date:
            return {}
        return out_data
    except:
        return {}


if __name__ == '__main__':
    in_data_with_timestamp = json.load(sys.argv[1])
    in_retrieval_file = sys.argv[2]
    granularity = sys.argv[3]  # session turn
    id2timestamp = {x['question_id']: x['timestamped_facts'] for x in in_data_with_timestamp}
    
    out_file = in_retrieval_file + '.timefiltered{}'.format(model if model == 'gpt-4o' else 'llama')
    in_retrieval_data = [json.loads(line) for line in open(in_retrieval_file).readlines()]

    out_data = []
    out_log = []

    metrics_before, metrics_after = [], []
    for entry in tqdm(in_retrieval_data):
        print('\n=======================================\n')
        
        event_date_to_sid = {}
        for sid, decomp_events in zip(entry['haystack_session_ids'], id2timestamp[entry['question_id']]):
            if not decomp_events:
                continue
            for event_entry in decomp_events:
                try:
                    if event_entry['date'] not in event_date_to_sid:
                        event_date_to_sid[event_entry['date']] = []
                    event_date_to_sid[event_entry['date']].append(sid)
                except:
                    continue
                
        # check if the question is time-sensitive and get the proposed timerange
        timerange = infer_time_range(entry['question'], entry['question_date'])
        out_log.append({
            'question': entry['question'],
            'question_date': entry['question_date'],
            'speculated_timerange': timerange
        })

        # print the original evaluation results
        ranked_items_original = entry['retrieval_results']['ranked_items']
        correct_docs = list(set([x['corpus_id'] for x in ranked_items_original if "answer" in x['corpus_id']]))
        corpus_ids = [x['corpus_id'] for x in ranked_items_original]
        rankings = [i for i in range(len(ranked_items_original))]
        cur_metrics = {}
        for k in [5, 10]:
            recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
            cur_metrics.update({
                #'recall_any@{}'.format(k): recall_any,
                'recall_all@{}'.format(k): round(recall_all, 4),
                'ndcg_any@{}'.format(k): round(ndcg_any, 4)
            })
        print(cur_metrics)
        metrics_before.append(cur_metrics)
            
        
        if not timerange:
            print('No date extracted:', entry['question'])
            out_data.append(entry)
            metrics_after.append(cur_metrics)
            continue
        else:
            print(json.dumps(out_log[-1], indent=4))
            
            # filter haystack
            filter_start_date = increment_date(timerange['start'], -2)
            filter_end_date = increment_date(timerange['end'], 2)
            sid_in_range, sid_not_in_range = [], []
            for date in event_date_to_sid:
                if date >= filter_start_date and date <= filter_end_date:
                    sid_in_range += event_date_to_sid[date]
            
            # we move the filtered out sessions to the end of the ranked items
            ranked_items_new_left, ranked_items_new_right = [], []
            for retrieved_entry in ranked_items_original:
                if granularity == 'session':
                    if retrieved_entry['corpus_id'] in sid_in_range:
                        ranked_items_new_left.append(retrieved_entry)
                    else:
                        ranked_items_new_right.append(retrieved_entry)
                elif granularity == 'turn':
                    if any([x in retrieved_entry['corpus_id'] for x in sid_in_range]):
                        ranked_items_new_left.append(retrieved_entry)
                    else:
                        ranked_items_new_right.append(retrieved_entry)
                else:
                    raise NotImplementedError

            ranked_items_new = ranked_items_new_left + ranked_items_new_right
            entry['retrieval_results']['ranked_items'] = ranked_items_new
            out_data.append(entry)
            
            # print the new evaluation results
            corpus_ids = [x['corpus_id'] for x in ranked_items_new]
            rankings = [i for i in range(len(ranked_items_new))]
            cur_metrics = {}
            for k in [5, 10]:
                recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
                cur_metrics.update({
                    #'recall_any@{}'.format(k): recall_any,
                    'recall_all@{}'.format(k): round(recall_all, 4),
                    'ndcg_any@{}'.format(k): round(ndcg_any, 4)
                })
            print(cur_metrics)
            metrics_after.append(cur_metrics)
            
    print('Metrics before:')
    print({round(np.mean([x[k] for x in metrics_before]), 4) for k in metrics_before[0]})
    print('Metrics after:')
    print({round(np.mean([x[k] for x in metrics_after]), 4) for k in metrics_after[0]})

    with open(out_file, 'w') as out_f:
        for entry in out_data:
            print(json.dumps(entry), file=out_f)
    print('Saved to', out_file)
