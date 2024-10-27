import os
import sys
import json
import random
from datetime import datetime, timedelta
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer


def format_date(date_str):
    parts = date_str.split('/')
    if len(parts) != 3:
        raise ValueError("Date must be in the format YYYY/MM/DD")

    year, month, day = parts

    if len(month) == 1:
        month = '0' + month
    if len(day) == 1:
        day = '0' + day

    return f"{year}/{month}/{day}"

def random_date(start_year, end_year, start_month=1, end_month=12, start_month_day=1, end_month_day=31):
    # Generate a random date within the specified range
    start_date = datetime(year=start_year, month=start_month, day=start_month_day)
    end_date = datetime(year=end_year, month=end_month, day=end_month_day)
    delta_days = (end_date - start_date).days
    random_days = random.randint(0, delta_days)
    random_date = start_date + timedelta(days=random_days)
    return random_date

def generate_random_dates_before(input_date_str, n, days=30):
    input_date = datetime.strptime(input_date_str, "%Y/%m/%d")
    # Get the date one month before the input date
    one_month_before = input_date - timedelta(days=days)
    # Generate n random dates within the range
    random_dates = [
        one_month_before + timedelta(days=random.randint(0, (input_date - one_month_before).days))
        for _ in range(n)
    ]
    # Sort the dates
    random_dates.sort()
    return [date.strftime("%Y/%m/%d") for date in random_dates]

def generate_random_dates_after(input_date_str, n, days=30):
    input_date = datetime.strptime(input_date_str, "%Y/%m/%d")
    # Get the date one month after the input date
    one_month_after = input_date + timedelta(days=days)
    # Generate n random dates within the range
    random_dates = [
        input_date + timedelta(days=random.randint(1, (one_month_after - input_date).days))
        for _ in range(n)
    ]
    # Sort the dates
    random_dates.sort()    
    return [date.strftime("%Y/%m/%d") for date in random_dates]

def generate_random_dates_in_range(start_date_str, end_date_str, n):
    start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
    end_date = datetime.strptime(end_date_str, "%Y/%m/%d")
    # Calculate the number of days in the range
    delta_days = (end_date - start_date).days
    if delta_days < 0:
        raise ValueError("End date must be after start date")
    # Generate n random dates within the range
    random_dates = [
        start_date + timedelta(days=random.randint(0, delta_days))
        for _ in range(n)
    ]
    # Sort the dates
    random_dates.sort()
    return [date.strftime("%Y/%m/%d") for date in random_dates]

def get_random_same_day_timestamps(n, base_date=None):
    if base_date is None:
        base_date = random_date(2023, 2023, 5, 5, 20, 30).strftime("%Y/%m/%d")
    base_date = datetime.strptime(base_date, "%Y/%m/%d")

    random_times = []
    for _ in range(n):
        # Generate a random number of seconds since the start of the day
        random_seconds = random.randint(0, 86399)  # 86399 seconds in a day
        random_time = base_date + timedelta(seconds=random_seconds)
        random_times.append(random_time)
    random_times.sort()
    formatted_times = [time.strftime("%Y/%m/%d (%a) %H:%M") for time in random_times]
    return formatted_times


# parameters
if len(sys.argv) != 6:
    print('Usage: python sample_haystack_and_timestamp.py task n_questions min_n_haystack_filler max_n_haystack_filler enforce_json_length')
    exit()
task = sys.argv[1] #'single_hop'
n_questions = int(sys.argv[2])   # 25
min_n_haystack_filler, max_n_haystack_filler = int(sys.argv[3]), int(sys.argv[4])   # 10, 10
haystack_source_ratio_user, haystack_source_ratio_sharegpt, haystack_source_ratio_ultrachat = 0.5, 0.25, 0.25
assert haystack_source_ratio_user + haystack_source_ratio_sharegpt + haystack_source_ratio_ultrachat == 1

prefix = '202410_custom_haystack1'
ref_model_name = 'meta-llama/Llama-3.1-8B-Instruct'
try:
    enforce_json_length = int(sys.argv[5])
except:
    enforce_json_length = None
suffix = 'user{}sharegpt{}ultrachat{}'.format(haystack_source_ratio_user, haystack_source_ratio_sharegpt, haystack_source_ratio_ultrachat)
if enforce_json_length is not None:
    print('Warning: enforce_json_length is not None, will pack the haystack to {} tokens according to {}'.format(enforce_json_length, ref_model_name))
    out_file = '{}_{}_{}-{}haysess_{}_enflen{}.json'.format(prefix, task, min_n_haystack_filler, max_n_haystack_filler, suffix, enforce_json_length)
else:
    out_file = '{}_{}_{}-{}haysess_{}.json'.format(prefix, task, min_n_haystack_filler, max_n_haystack_filler, suffix)

# database
database_dir = os.getcwd()
attribute_db = json.load(open(database_dir + '/1_attr_bg/data_1_attr_bg.json'))
bgid_to_attribute = {}
for a in attribute_db:
    for bg_entry in a['backgrounds']:
        bgid_to_attribute[bg_entry['background_id']] = bg_entry['attribute_id']
question_db_file = database_dir + '/2_questions/0822_all_500_questions_final_v3.json'
question_db = json.load(open(question_db_file))
print('Current question db:', question_db_file)
qid2attid = {x['question_id']: bgid_to_attribute[x['background_id']] if x['background_id'] in bgid_to_attribute else x['background_id'] for x in question_db}

filler_db = json.load(open(database_dir + '/5_filler_sess/data_5_filler_sess.json'))
filler_idx = {x['session_id']: x for x in filler_db}
session_db = json.load(open(database_dir + '/6_session_cache/data_6_session_cache.json'))
sess_idx = {x['session_id']: x for x in session_db}

haystack_source_sessions_user_raw = [x for x in json.load(open(database_dir + '/6_session_cache/data_6_session_cache.json'))]
haystack_source_sessions_user_all = []
for x in haystack_source_sessions_user_raw:
    if 'session' in x:
        haystack_source_sessions_user_all.append(x)
    elif 'sessions' in x:
        for i, s in enumerate(x['sessions']):
            out_entry = deepcopy(x)
            out_entry['session'] = s
            out_entry['session_id'] += f'_{i+1}'
            haystack_source_sessions_user_all.append(out_entry)
    elif 'session_1' in x and 'session_2' in x:
        out_entry = deepcopy(x)
        out_entry['session'] = x['session_1']
        out_entry['session_id'] += f'_1'
        haystack_source_sessions_user_all.append(out_entry)
        out_entry = deepcopy(x)
        out_entry['session'] = x['session_2']
        out_entry['session_id'] += f'_2'
        haystack_source_sessions_user_all.append(out_entry)
    elif 'old_session' in x and 'new_session' in x:
        out_entry = deepcopy(x)
        out_entry['session'] = x['old_session']
        out_entry['session_id'] += f'_1'
        haystack_source_sessions_user_all.append(out_entry)
        out_entry = deepcopy(x)
        out_entry['session'] = x['new_session']
        out_entry['session_id'] += f'_2'
        haystack_source_sessions_user_all.append(out_entry)
                
haystack_source_sessions_sharegpt = [x for x in filler_db if 'sharegpt' in x['session_id']]
haystack_source_sessions_ultrachat = [x for x in filler_db if 'ultrachat' in x['session_id']]
print('Haystack source: {} user, {} sharegpt, {} ultrachat'.format(len(haystack_source_sessions_user_all), len(haystack_source_sessions_sharegpt), len(haystack_source_sessions_ultrachat)))

# sample questions
if task != 'assistant_previnfo':
    question_candidates = []
    for x in question_db:
        if x['question_type'] != task:
            continue
        if not [y for y in x['sessions'] if 'human_valid_label' in sess_idx[y['session_id']] and sess_idx[y['session_id']]['human_valid_label']]:
            continue
        question_candidates.append(x)
else:
    question_candidates = [x for x in question_db if x['question_type'] == task and x['sessions']]
questions = question_candidates[:n_questions]

# sample haystack for each question
out_data = []
tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
out_token_counts = []
for question_entry in tqdm(questions):
    # get the subset of user sessions that do not have conflicting attributes
    haystack_source_sessions_user_cur_subset = []
    for x in haystack_source_sessions_user_all:
        if 'question_id' not in x or x['question_id'] not in qid2attid or qid2attid[x['question_id']] != qid2attid[question_entry['question_id']]:
            haystack_source_sessions_user_cur_subset.append(x)
    
    # sample haystack
    cur_n_haystack = random.choice(list(range(min_n_haystack_filler, max_n_haystack_filler+1)))
    haystack = []
    for _ in range(cur_n_haystack):
        rand_float = random.random()
        if rand_float < haystack_source_ratio_user:
            haystack.append(random.choice(haystack_source_sessions_user_cur_subset))
        elif rand_float > haystack_source_ratio_user + haystack_source_ratio_sharegpt:
            haystack.append(random.choice(haystack_source_sessions_ultrachat))
        else:
            haystack.append(random.choice(haystack_source_sessions_sharegpt))
    haystack = [{'session_id': x['session_id'], 'session': x['session']} for x in haystack]
    # clean up the potential has_answer labels in the turns of non-answer sessions
    haystack = [
        {'session_id': x['session_id'], 'session': [{'role': y['role'], 'content': y['content']} for y in x['session']]}
        for x in haystack
    ]
    
    # order sessions
    if task in ['single_hop', 'implicit_preference', 'implicit_preference_v2']:
        answer_session = None
        for sess_entry in question_entry['sessions']:
            if sess_entry['style'] == 'neutral' and 'human_valid_label' in sess_idx[sess_entry['session_id']] and sess_idx[sess_entry['session_id']]['human_valid_label']:
                answer_session = sess_idx[sess_entry['session_id']]
                break
        assert answer_session is not None
        haystack.append({'session_id': 'answer_' + answer_session['session_id'], 'session': answer_session['session']})
        random.shuffle(haystack)
    elif task in ['assistant_previnfo']:
        answer_session = None
        for sess_entry in question_entry['sessions']:
            if sess_entry['style'] == 'neutral':
                answer_session = filler_idx[sess_entry['session_id']]
                break
        assert answer_session is not None
        haystack.append({'session_id': 'answer_' + answer_session['session_id'], 'session': answer_session['session']})
        random.shuffle(haystack)
    elif task == 'two_hop':
        answer_session = None
        for sess_entry in question_entry['sessions']:
            if sess_entry['style'] == 'neutral' and sess_idx[sess_entry['session_id']]['human_valid_label']:
                answer_session = sess_idx[sess_entry['session_id']]
                break
        assert answer_session is not None
        haystack.append({'session_id': 'answer_' + answer_session['session_id'] + '_1', 'session': answer_session['session_1']})
        haystack.append({'session_id': 'answer_' + answer_session['session_id'] + '_2', 'session': answer_session['session_2']})
        random.shuffle(haystack)
    elif task in ['multi_session_synthesis', 'temp_reasoning_explicit']:
        answer_session = None
        for sess_entry in question_entry['sessions']:
            if sess_entry['style'] == 'neutral' and sess_idx[sess_entry['session_id']]['human_valid_label']:
                answer_session = sess_idx[sess_entry['session_id']]
                break
        assert answer_session is not None
        for i_sess, sess_entry in enumerate(answer_session['sessions']):
            haystack.append({'session_id': 'answer_' + answer_session['session_id'] + '_{}'.format(i_sess+1), 'session': sess_entry})
        random.shuffle(haystack)
    elif task == 'knowledge_update':
        answer_session = None
        for sess_entry in question_entry['sessions']:
            if sess_entry['style'] == 'neutral' and sess_idx[sess_entry['session_id']]['human_valid_label']:
                answer_session = sess_idx[sess_entry['session_id']]
                break
        assert answer_session is not None

        split_loc = random.choice(list(range(0, len(haystack)+1)))
        left_stack = haystack[:split_loc]
        left_stack.append({'session_id': 'answer_' + answer_session['session_id'] + '_1', 'session': answer_session['session_old']})
        random.shuffle(left_stack)
        right_stack = haystack[split_loc:]
        right_stack.append({'session_id': 'answer_' + answer_session['session_id'] + '_2', 'session': answer_session['session_new']})
        random.shuffle(right_stack)
        haystack = left_stack + right_stack
    elif task == 'temp_reasoning_implicit':
        answer_session = None
        # print(question_entry)
        for sess_entry in question_entry['sessions']:
            if sess_entry['style'] == 'neutral' and sess_idx[sess_entry['session_id']]['human_valid_label']:
                answer_session = sess_idx[sess_entry['session_id']]
                break
        assert answer_session is not None

        haystack += ['ans_placeholder' for _ in range(len(answer_session['sessions']))] 
        random.shuffle(haystack)
        cur_ans_sess_id = 0
        for i_sess in range(len(haystack)):
            if haystack[i_sess] != 'ans_placeholder':
                continue
            else:
                haystack[i_sess] = {
                    'session_id': 'answer_' + answer_session['session_id'] + '_{}'.format(cur_ans_sess_id + 1),
                    'session': answer_session['sessions'][cur_ans_sess_id]
                }
                cur_ans_sess_id += 1
    else:
        raise NotImplementedError

    # resolve timestamp constraints
    if task in ['two_hop', 'multi_session_synthesis', 'single_hop', 'implicit_preference', 'implicit_preference_v2', 'assistant_previnfo']:
        # dates = a series of random time on the same day
        if 'unified_date' in question_entry['question_content']:
            dates = get_random_same_day_timestamps(len(haystack)+1, base_date=question_entry['question_content']['unified_date'])
        else:
            dates = sorted([get_random_same_day_timestamps(1)[0] for _ in range(len(haystack)+1)])
        for i in range(len(haystack)):
            haystack[i]['date'] = dates[i]
        question_date = question_entry['question_content']['question_date'] if 'question_date' in question_entry['question_content'] else dates[-1]
    elif task == 'temp_reasoning_explicit':
        if 'unified_date' in question_entry['question_content']:
            unified_base_date = question_entry['question_content']['unified_date']
        else:
            unified_base_date = random_date(2023, 2023, 5, 5, 20, 30).strftime("%Y/%m/%d")
        dates = [get_random_same_day_timestamps(1, base_date=unified_base_date)[0] for _ in range(len(haystack)+1)]
        for i in range(len(haystack)):
            haystack[i]['date'] = dates[i]
        # question_date = dates[-1]
        question_date = question_entry['question_content']['question_date'] if 'question_date' in question_entry['question_content'] else dates[-1]
    elif task == 'temp_reasoning_implicit':
        for i in range(len(haystack)):
            if 'answer_' in haystack[i]['session_id']:
                haystack[i]['date'] = question_entry['question_content']['facts'][int(haystack[i]['session_id'][-1])-1]['date']
                # special process
                haystack[i]['date'] = format_date(haystack[i]['date'])
        # hack
        special_session_reordered = sorted([x for x in haystack if 'answer_' in x['session_id']], key=lambda x: x['date'])
        spec_session_count = 0
        for i in range(len(haystack)):
            if 'answer_' in haystack[i]['session_id']:
                haystack[i] = special_session_reordered[spec_session_count]
                spec_session_count += 1
                
        left_date, right_date = None, None
        haystack_date_cache = []
        for i in range(len(haystack)):
            j = i + 1
            while j < len(haystack):
                if 'answer_' in haystack[j]['session_id']:
                    right_date = haystack[j]['date']
                    break
                j += 1
            if 'answer_' in haystack[i]['session_id']:
                left_date = haystack[i]['date']
                haystack_date_cache.append(haystack[i]['date'])
            else:
                if left_date is None:
                    haystack[i]['date'] = generate_random_dates_before(right_date, 1)[0]
                elif right_date is None:
                    haystack[i]['date'] = generate_random_dates_after(left_date, 1)[0]
                else:
                    haystack[i]['date'] = generate_random_dates_in_range(left_date, right_date, 1)[0]
                haystack_date_cache.append(haystack[i]['date'])
        haystack_date_cache.sort()
        for i in range(len(haystack)):
            haystack[i]['date'] = haystack_date_cache[i]
            haystack[i]['date'] = get_random_same_day_timestamps(1, base_date=haystack[i]['date'])[0]

        question_date = get_random_same_day_timestamps(1, base_date=question_entry['question_content']['question_date'])[0]
        # hack
        if question_date < haystack[i]['date']:
            question_date, haystack[i]['date'] = haystack[i]['date'], question_date
    elif task == 'knowledge_update':
        if 'temporal_constraint' in question_entry['question_content']:
            date_old = question_entry['question_content']['temporal_constraint']['fact_1_date']
            date_new = question_entry['question_content']['temporal_constraint']['fact_2_date']
        else:
            date_old = random_date(2023, 2023, 5, 5, 20, 30).strftime("%Y/%m/%d")
            date_new = random_date(2023, 2023, 5, 5, 20, 30).strftime("%Y/%m/%d")
            if date_new < date_old:
                date_old, date_new = date_new, date_old
        if 'question_date' in question_entry['question_content']:
            question_date = question_entry['question_content']['question_date']
        else:
            question_date = generate_random_dates_after(date_new, 1)[0]
        dates = []
        left_hit, right_hit = False, False
        for hs_entry in haystack:
            if hs_entry['session_id'] == 'answer_' + answer_session['session_id'] + '_1':
                left_hit = True
                dates.append(date_old)
            elif hs_entry['session_id'] == 'answer_' + answer_session['session_id'] + '_2':
                right_hit = True
                dates.append(date_new)
            else:
                if not left_hit:
                    dates += generate_random_dates_before(date_old, 1)
                elif not right_hit:
                    dates += generate_random_dates_in_range(date_old, date_new, 1)
                else:
                    dates += generate_random_dates_in_range(date_new, question_date, 1)
                    # dates += generate_random_dates_after(date_new, 1)
        dates.sort()
        dates.append(question_date)
        dates = [get_random_same_day_timestamps(1, base_date=x)[0] for x in dates]
        for i in range(len(haystack)):
            haystack[i]['date'] = dates[i]
        question_date = dates[-1]
        
    else:
        raise NotImplementedError

    # output
    out_entry = {
        'question_id': question_entry['question_id'],
        'question_type': question_entry['question_type'],
        'question': question_entry['question_content']['question'],
        'answer': question_entry['question_content']['answer'],
        'question_date': question_date,
        'haystack_dates': [x['date'] for x in haystack],
        'haystack_session_ids': [x['session_id'] for x in haystack],
        'haystack_sessions': [x['session'] for x in haystack],
        'answer_session_ids': [x['session_id'] for x in haystack if 'answer' in x['session_id']]
    }

    if enforce_json_length:
        while len(tokenizer.encode(json.dumps(out_entry['haystack_sessions']))) > enforce_json_length:
            # randomly drop one non-answer session
            filler_indices = [i for i, s in enumerate(out_entry['haystack_session_ids']) if 'answer_' not in s]
            remove_idx = random.choice(filler_indices)
            out_entry['haystack_dates'] = out_entry['haystack_dates'][:remove_idx] + out_entry['haystack_dates'][remove_idx+1:]
            out_entry['haystack_session_ids'] = out_entry['haystack_session_ids'][:remove_idx] + out_entry['haystack_session_ids'][remove_idx+1:]
            out_entry['haystack_sessions'] = out_entry['haystack_sessions'][:remove_idx] + out_entry['haystack_sessions'][remove_idx+1:]

    out_token_counts.append(len(tokenizer.encode(json.dumps(out_entry['haystack_sessions']))))
    out_data.append(out_entry)


# dump
json.dump(out_data, open(out_file, 'w'), indent=4)
print('Max token count {}; Min token count {}; Mean token count {}'.format(max(out_token_counts), min(out_token_counts), round(sum(out_token_counts)/len(out_token_counts), 2)))