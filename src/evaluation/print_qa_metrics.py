import sys
import json
import numpy as np


if len(sys.argv) != 3:
    print('Usage: python print_qa_metrics.py in_file ref_file')
    exit()

in_file = sys.argv[1]
ref_file = sys.argv[2]
in_data = [json.loads(line) for line in open(in_file).readlines()]
ref_data = json.load(open(ref_file))
ref_data = {x['question_id']: x for x in ref_data}

# Metric models supported by evaluate_qa.py (see model_zoo there).
# We do not hardcode a single model, but we do require all entries in the
# log to be judged by the *same* model so the aggregated numbers are coherent.
eval_models = set(entry['autoeval_label']['model'] for entry in in_data)
assert len(eval_models) == 1, (
    'All entries must be evaluated by the same metric model, but found: '
    '{}'.format(sorted(eval_models))
)

all_acc, task_acc, abstention_acc = [], [], []
type2acc = {t: [] for t in ['single-session-user', 'single-session-preference', 'single-session-assistant', 'multi-session', 'temporal-reasoning', 'knowledge-update']}
for entry in in_data:
    ref_entry = ref_data[entry['question_id']]
    type2acc[ref_entry['question_type']].append(1 if entry['autoeval_label']['label'] else 0)
    if '_abs' in entry['question_id']:
        abstention_acc.append(1 if entry['autoeval_label']['label'] else 0)

print('\nEvaluation results by task:')
for k, v in type2acc.items():
    print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))
    all_acc += v
    task_acc.append(np.mean(v))

print('\nTask-averaged Accuracy:', round(np.mean(task_acc), 4))
print('Overall Accuracy:', round(np.mean(all_acc), 4))
print('Abstention Accuracy:', round(np.mean(abstention_acc), 4), f'({len(abstention_acc)})')
