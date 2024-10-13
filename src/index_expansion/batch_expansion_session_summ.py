import os
import json
from tqdm import tqdm
from openai import OpenAI
import openai
import backoff

@backoff.on_exception(backoff.constant, (openai.RateLimitError), 
                      interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


client = OpenAI(
    api_key="empty",
    base_url="http://localhost:8001/v1",
)


def summarize_session(sess_entry, model_name):
    # memorybank prompt
    # summarization_prompt = "Below is a transcript of a conversation between a human user and an AI assistant. Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately. Dialogue content:\n"
    summarization_prompt = "Below is a transcript of a conversation between a human user and an AI assistant. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information. In your summary, focus more on what the user mentioned or asked for. Dialogue content:\n"
    for turn_entry in sess_entry:
        summarization_prompt += f"\n{turn_entry['role']}：{turn_entry['content']}"
    summarization_prompt += '\n\nYour summary (be concise):'
    # print(summarization_prompt)

    kwargs = {
        'model': model_name,
        'messages':[
            {"role": "user", "content": summarization_prompt}
        ],
        'n': 1,
        'temperature': 0,
        'max_tokens': 500
    }
    completion = chat_completions_with_backoff(client,**kwargs) 
    return completion.choices[0].message.content.strip()


if __name__ == '__main__':
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    
    
    in_file = '/home/diwu/ralm/long-mem-benchmark/data/userinfo_v2/5_filler_sess/data_5_filler_sess.json'
    # in_file = '/home/diwu/ralm/long-mem-benchmark/data/userinfo_v2/6_session_cache/data_6_session_cache.json'
    cache_file = '/local2/diwu/long-mem-benchmark/index_expansion_logs/' + in_file.split('/')[-1] + '.session-summ.json'
    
    if os.path.isfile(cache_file):
        data = json.load(open(cache_file))
        print('Loaded:', cache_file)
    else:
        data = {}

    in_data = json.load(open(in_file))

    todo_sessions = []
    for entry in in_data:
        if 'session' in entry:
            todo_sessions.append((entry['session_id'], entry['session']))
        elif 'sessions' in entry:
            for i, s in enumerate(entry['sessions']):
                todo_sessions.append((entry['session_id'] + f'_{i+1}', s))
        elif 'session_1' in entry and 'session_2' in entry:
            todo_sessions.append((entry['session_id'] + '_1', entry['session_1']))
            todo_sessions.append((entry['session_id'] + '_2', entry['session_2']))
        elif 'old_session' in entry and 'new_session' in entry:
            todo_sessions.append((entry['session_id'] + '_1', entry['old_session']))
            todo_sessions.append((entry['session_id'] + '_2', entry['new_session']))

    todo_sessions = [(i, s) for i, s in todo_sessions if i not in data]
    for i, entry in tqdm(todo_sessions):
        expansion = summarize_session(entry, model_name)
        data[i] = expansion
        print({i: expansion})
        
    json.dump(data, open(cache_file, 'w'))
