import sys
import json
import numpy as np


if len(sys.argv) != 2:
    print('Usage: python print_retrieval_metrics.py in_file')
    exit()

in_file = sys.argv[1]
in_data = [json.loads(line) for line in open(in_file).readlines()]

task2type = {
    'single_hop': 'single_needle',
    'assistant_previnfo': 'single_needle',
    'two_hop': 'multi_session_synthesis',
    'multi_session_synthesis': 'multi_session_synthesis',
    'knowledge_update': 'knowledge_update',
    'temp_reasoning_explicit': 'temporal_reasoning',
    'temp_reasoning_implicit': 'temporal_reasoning',
    'implicit_preference_v2': 'implicit_preference_v2'
}
type2acc = {t: [] for t in set(list(task2type.values()))}

all_metrics = []
for entry in in_data:
    all_metrics.append(entry['retrieval_results']['metrics'])

sess_metric_names = ['recall_all@5', 'ndcg_any@5', 'recall_all@10', 'ndcg_any@10']
print('Session-level metrics:')
try:
    print(', '.join(['\t{} = {}'.format(name, round(np.mean([x['session'][name] for x in all_metrics]), 4)) for name in sess_metric_names]))
except:
    pass

turn_metric_names = ['recall_all@5', 'ndcg_any@5', 'recall_all@10', 'ndcg_any@10', 'recall_all@50', 'ndcg_any@50']
print('Turn-level metrics:')
try:
    print(', '.join(['\t{} = {}'.format(name, round(np.mean([x['turn'][name] for x in all_metrics]), 4)) for name in turn_metric_names]))
except:
    pass


# ref_entry = ref_data[entry['question_id']]
# assert entry['autoeval_label']['model'] == 'gpt-4o-2024-08-06'
# type2acc[task2type[ref_entry['question_type']]].append(1 if entry['autoeval_label']['label'] else 0)

# all_acc = []
# task_acc = []
# print('\nEvaluation results by task:')
# for k, v in type2acc.items():
#     print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))
#     all_acc += v
#     task_acc.append(np.mean(v))

# print('\nTask-averaged Accuracy:', round(np.mean(task_acc), 4))

# print('\nOverall Accuracy:', round(np.mean(all_acc), 4))
