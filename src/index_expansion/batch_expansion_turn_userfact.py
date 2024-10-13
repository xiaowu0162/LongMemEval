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


def extract_round_userfact(sess_entry, model_name, examples=None):
    system_prompt = "You will be given a message from a human user to an AI assistant. Extract all the personal information, life events, experience, and preferences related to the user. Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, or dates. State each piece of information in a simple sentence. Put these sentences in a json list, each element being a standalone personal fact about the user. Minimize the coreference across the facts, e.g., replace pronouns with actual entities. If there is no specific events, personal information, or preference mentioned, just generate an empty list."
    
    user_prompt = "Human user message:\n{}\n\nPersonal facts about the user (a list of strings in json format; do not generate anything else):"

    dialogue_string = ""
    for turn_entry in sess_entry:
        if turn_entry['role'] == 'user':
            dialogue_string += f"\n{turn_entry['role']}：{turn_entry['content']}"

    print([dialogue_string])

    summarization_prompt = user_prompt.format(dialogue_string)
    if examples is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summarization_prompt}
        ]
    else:
        messages = [{"role": "system", "content": system_prompt}]
        for example_input_dialogue_string, example_output in examples:
            messages += [
               {"role": "user", "content": user_prompt.format(example_input_dialogue_string)},
               {"role": "assistant", "content": example_output}
            ]
        messages += [{"role": "user", "content": summarization_prompt}]

    kwargs = {
        'model': model_name,
        'messages': messages,
        'n': 1,
        'temperature': 0,
        'max_tokens': 2000
    }
    completion = chat_completions_with_backoff(client,**kwargs)
    try:
        out_string = completion.choices[0].message.content.strip()
        out_string = out_string.replace('```json', '')
        out_string = out_string.replace('```', '')
        return json.loads(out_string.strip())
    except:
        #print(completion.choices[0].message.content.strip())
        #exit()
        return None


if __name__ == '__main__':
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    # model_name = 'gpt-4o'

    mode = 'ICL'   # zero-shot, ICL
    assert mode in ['zero-shot', 'ICL']
    
    # in_file = '/home/diwu/ralm/long-mem-benchmark/data/userinfo_v2/5_filler_sess/data_5_filler_sess.json.shard2'
    in_file = '/home/diwu/ralm/long-mem-benchmark/data/userinfo_v2/6_session_cache/data_6_session_cache.json.shard2'
    
    # cache_file = in_file + f'.session-userfact.{mode}.json'
    cache_file = '/local2/diwu/long-mem-benchmark/index_expansion_logs/' + in_file.split('/')[-1] + f'.turn-userfact.{mode}.json'
    
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
        for i_turn in range(len(entry)):
            if entry[i_turn]['role'] == 'user':
                if mode == 'zero-shot':
                    expansion = extract_round_userfact([entry[i_turn]], model_name, examples=None)
                else:
                    examples = [
                        ('\nuser：What notable religious sites can visitors explore in Udine?', json.dumps([])),   # from sharegpt/ultrachat
                        ("\nuser：Wow, I never thought to add cheese to my grilled asparagus! I think I'll try the goat cheese option.", json.dumps(['The user has never added cheese to grilled asparagus before.', 'The user is considering trying goat cheese with grilled asparagus.'])),   # from sharegpt/ultrachat
                        ("\nuser：I don't know, Lucky Buns and Duke's Grocery sounds a little too fancy for a burger joint. I just want a good old-fashioned burger.", json.dumps(["The user thinks Lucky Buns and Duke's Grocery sound too fancy for a burger joint.", 'The user prefers a good old-fashioned burger.'])),   # from sharegpt/ultrachat
                        ("\nuser：That's really fascinating! It sounds like technology is a game-changer in the architecture industry. How do architects keep up with all these new tools and software?", json.dumps([])),   # from sharegpt/ultrachat
                        ("\nuser：Wow, those are some really inspiring examples. It's great to see how rejection can actually lead to success in the end. Do you have any tips for how to handle rejection in a positive way?", json.dumps([])),   # from sharegpt/ultrachat
                        ("\nuser：I'm looking to buy a house and I'm not sure how to calculate my mortgage payments. Can you help me with that? By the way, I recently got pre-approved for a mortgage and the lender said I can borrow up to $350,000.", json.dumps(['The user is looking to buy a house.', 'The user recently got pre-approved for a mortgage.', "The user's lender said the user can borrow up to $350,000 for the mortgage."])),   # from userinfo
                        ("\nuser：I'm 32, so I'm in my 30s. I'd say my skin type is normal, but it's been looking a bit dull lately. My main concerns are fine lines and wrinkles, especially around my eyes. I'm open to trying out different products, but I'd prefer something not too expensive.", json.dumps(['The user is 32 years old.', "The user's skin type is normal.", "The user's skin has been looking a bit dull lately.", "The user's main skin concerns are fine lines and wrinkles.", 'The user is especially concerned about fine lines and wrinkles around their eyes.', 'The user is open to trying out different skincare products.', 'The user prefers skincare products that are not too expensive.'])),   # from userinfo
                        ("\nuser：I frequent Amazon, eBay, ASOS, and some online thrift stores like ThredUp. I recently got into online thrift shopping, and I actually just bought another pair of jeans from ThredUp on February 12th for $30.", json.dumps(['The user frequents Amazon.', 'The user frequents eBay.', 'The user frequents ASOS.', 'The user frequents online thrift stores like ThredUp.', 'The user recently got into online thrift shopping.', 'The user bought a pair of jeans from ThredUp on February 12th for $30.'])),   # from userinfo
                        ("\nuser：I think I'll go with a 20,000mAh power bank for now, as it should be sufficient for my daily needs. By the way, I've been really enjoying my Sony WH-1000XM4 headphones, and the noise-cancelling feature has been a game-changer for my daily commute.", json.dumps(['The user plans to use a 20,000mAh power bank.', 'The user believes a 20,000mAh power bank will be sufficient for daily needs.', 'The user enjoys using Sony WH-1000XM4 headphones.', 'The user finds the noise-cancelling feature of the Sony WH-1000XM4 headphones to be a game-changer.', 'The user uses the noise-cancelling feature of the Sony WH-1000XM4 headphones during daily commutes.'])),   # from userinfo
                        ("nuser：I like the idea of adding a special touch to the photo album. Do you think a decorative box or a personalized message would be a better addition?", json.dumps([])),   # from userinfo
                    ]
                    expansion = extract_round_userfact([entry[i_turn]], model_name, examples=examples)
                data[i + f'_{i_turn+1}'] = expansion
                print({i + f'_{i_turn+1}': expansion})
        
    json.dump(data, open(cache_file, 'w'))
