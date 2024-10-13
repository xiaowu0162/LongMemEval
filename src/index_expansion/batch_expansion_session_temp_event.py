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
    base_url="http://localhost:8002/v1",
)

def extract_session_userfact(sess_date, sess_entry, model_name, examples=None):
    system_prompt = "You will be given a list of messages from a human user to an AI assistant, as well as the time the conversation took place. Extract all events related to the user as long as its date is specified or could be inferred. If the time some event took place cannot be inferred, do not extract that event. Return the events in a json list where each item contains two fields: \"date\" and \"event\". Write date in the form YYYY/MM/DD. If there is no specific event, just write an empty list."
    
    user_prompt = "Conversation date: {}\nHuman user messages:\n{}\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):"

    dialogue_string = ""
    for turn_entry in sess_entry:
        if turn_entry['role'] == 'user':
            dialogue_string += f"\n{turn_entry['role']}: {turn_entry['content']}"

    summarization_prompt = user_prompt.format(sess_date, dialogue_string)
    if examples is None:
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summarization_prompt}
            ]
    else:
        messages = [{"role": "system", "content": system_prompt}]
        for example_input_dialogue_string, example_output in examples:
            messages += [
                {"role": "user", "content": example_input_dialogue_string},
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
        out_string = out_string.replace('```', '').strip()
        print([json.dumps(json.loads(out_string))])
        return json.loads(out_string.strip())
    except:
        return None


if __name__ == '__main__':
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    # model_name = 'gpt-4o'

    mode = 'ICL'   # zero-shot, ICL
    assert mode in ['zero-shot', 'ICL']
    
    # in_file = '/home/diwu/ralm/long-mem-benchmark/data/userinfo_v2/6_session_cache/active_workspace/0918_knowledge_update_1000_for_timestamp_validation.json'
    in_file = '/home/diwu/ralm/long-mem-benchmark/data/userinfo_v2/7_haystacks/0826_all_questions_v3_with_sessions_115ktoks_haystack_user0.5sharegpt0.25ultrachat0.25.json.idresolved'
    # in_file = '/local2/diwu/long-mem-benchmark/final_haystacks/0826_all_questions_v3_with_sessions_500sess_haystack_user0.5sharegpt0.25ultrachat0.25.json.idresolved.cleaned.shard1.json'
    
    # cache_file = in_file + f'.session-timedfacts.{mode}.json'
    cache_file = '/local2/diwu/long-mem-benchmark/index_expansion_logs/' + in_file.split('/')[-1] + f'.session-userfact.{mode}.json'
    
    if os.path.isfile(cache_file):
        data = json.load(open(cache_file))
        print('Loaded:', cache_file)
    else:
        data = {}

    in_data = json.load(open(in_file))

    data = []
    for entry in tqdm(in_data):
        todo_sessions = entry['haystack_sessions']
        cur_expansions = []
        for i, (cur_date, sess_entry) in enumerate(zip(entry['haystack_dates'], todo_sessions)):
            if mode == 'zero-shot':
                expansion = extract_session_userfact(cur_date, sess_entry, model_name, examples=None)
            else:
                examples = [
                    ("Conversation date: 2023/05/29 (Mon) 10:53\nHuman user messages:\n\nuser: Describe the social hierarchies of pack animals like wolves and how they communicate through body language.\nuser: It's fascinating how wolves communicate with each other through body language and vocalizations. Do they ever show affection towards each other in this way?\nuser: It's really interesting how much body language and vocalizations can convey! Do other animals communicate in similar ways?\nuser: Wow, it's amazing how much animals can communicate without even using words. Do you have any examples of animals that communicate through smell?\nuser: It's amazing how much animals can communicate. Are there any animals that use touch to communicate with each other?\nuser: I had no idea that cats use touch to communicate so much! That's really interesting.\nuser: Wow, I had no idea cats were such communicators! Speaking of touch, do you know anything about how dolphins communicate with touch?\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[]'),
                    ("Conversation date: 2023/05/20 (Sat) 02:21\nHuman user messages:\n\nuser: The farmer needs to transport a fox, a chicken, and some grain across a river using a boat. The fox cannot be left alone with the chicken, and the chicken cannot be left alone with the grain. The boat can only hold one item at a time, and the river is too dangerous to cross multiple times. Can you help the farmer transport all three items across the river without any of them getting eaten? Remember, strategic thinking and planning are key to solving this puzzle. If you're stuck, try thinking about how you would solve the puzzle yourself, and use that as a starting point. Be careful not to leave the chicken alone with the fox, or the chicken and the grain alone together, as this will result in a failed solution. Good luck!\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[]'),
                    ("Conversation date: 2023/05/24 (Wed) 13:03\nHuman user messages:\n\nuser: How can I make tender and flavorful chicken breast for my weeknight meals, and what cooking methods should I avoid using to prevent dryness?\nuser: I think I'll try the brine solution and pounding the chicken breasts next time. Do you have any marinade recipes that you recommend?\nuser: These marinade recipes sound great! I think I'll try the soy ginger marinade first.\nuser: I just tried the soy ginger marinade and it was amazing! Do you have any suggestions for side dishes to pair with the chicken?\nuser: These side dish suggestions all sound great! I think I'll try roasting some vegetables to go with my chicken next time.\nuser: Hey, do you have any suggestions for a dessert that could go well with the chicken and roasted vegetables? Maybe something light and refreshing?\nuser: Hmm, I think I want to try making some homemade fruit popsicles for dessert! Do you have any tips on how to make them turn out just right?\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[{"date": "2023/05/24", "event": "User tried the soy ginger marinade and found it amazing."}]'),
                    ("Conversation date: 2023/05/25 (Thu) 09:47\nHuman user messages:\n\nuser: I'm looking for some new healthy snack ideas, something easy to grab and go.\nuser: I'm obsessed with hummus too! Do you have any new hummus recipes I can try?\nuser: I've been making a big batch of hummus every weekend, and I'd love to try some new flavors. Do you have any recipes that use roasted garlic? I've heard it adds a deep, nutty flavor to hummus.\nuser: Can you give me some suggestions for healthy snack bars that I can buy at the store?\nuser: I've tried RXBAR and Quest Bar before, but I'm interested in trying some new options. Can you tell me more about Kind Bar and LaraBar? What are some of their popular flavors and what makes them stand out from other healthy snack bars?\nuser: I've been meaning to try Kind Bar's Fruit & Nut flavor, and LaraBar's Coconut Cream Pie flavor sounds intriguing. Do you have any recommendations for other healthy snack options that are similar to these bars? Maybe something crunchy or crispy?\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[]'),
                    ('Conversation date: 2023/05/26 (Fri) 04:41\nHuman user messages:\n\nuser: how do you calculate a sorcerors spell save dc\nuser: is it possible to maintain two spells requiring concentration concurrently\nuser: best spell to increase AC for a sorceror / monk\nuser: how is a monks ac calculated\nuser: which sorceror spells available at 5th level are the best for enhancing a monk damage output\nuser: how can a 5th level sorcerer cast the 2nd level druid spell flame blade\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):', '[]'),
                    ("Conversation date: 2023/05/30 (Tue) 21:40\nHuman user messages:\n\nuser: I just got my new Instant Pot and kitchen knives from the Amazon Prime Day sales, and I'm excited to start cooking with them. Can you give me some recipe ideas for a beginner like me?\nuser: Can you give me some tips on how to properly clean and maintain my new Instant Pot and kitchen knives? I want to make sure they last a long time and stay in good condition.\nuser: I'm also planning to cook more at home and reduce food waste. Can you give me some tips on meal planning and grocery shopping?\nuser: Can you give me some advice on how to organize my kitchen utensils and spices? I recently bought a new Instant Pot and kitchen knives, and my kitchen is feeling a bit cluttered.\nuser: I'd like to get some more ideas on how to organize my kitchen cabinets. Can you give me some suggestions on how to maximize the storage space in my cabinets?\nuser: I'd like to get some more ideas on how to organize my kitchen pantry. Can you give me some suggestions on how to maximize the storage space in my pantry?\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[{"date": "2023/05/30", "event": "Received new Instant Pot and kitchen knives from Amazon Prime Day sales"}]'),
                    ("Conversation date: 2023/07/30 (Tue) 09:50\nHuman user messages:\n\nuser: I'm trying to organize my jewelry collection and was wondering if you could help me create a list to keep track of everything. By the way, I just got a replacement pair of earrings, no price mentioned, and I want to make sure I add those to the list.\nuser: I'll start by adding my grandmother's pearl necklace to the list. It's an antique, worth $5,000, and I'm not sure when I acquired it since it's been passed down. Also, can you help me figure out how to track the repair history of my watches? I have a watch that needs a new leather strap and battery replacement, and I want to keep a record of when I get these things done.\nuser: I'll add my mother's locket to the list. It's an old locket and I'm not sure when I acquired it, but I wore it to my cousin's wedding on June 15th.\nuser: I also need to add the aquamarine ring I got on sale at 20% off to the list. I bought it on June 1st, but I don't know the original price.\nuser: I also found a single pearl earring while cleaning out my jewelry box on July 3rd. I have no idea where it came from or who it belonged to, but it's a nice little mystery to solve. Can I add it to the list as well?\nuser: I'd like to add my watches to the list. I have two watches that need attention: one needs a new leather strap and the other needs its batteries replaced. Can I add them to the list with their respective repair needs?\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[{"date": "2023/07/30", "event": "User got a replacement pair of earrings."}, {"date": "2023/06/01", "event": "User bought an aquamarine ring on sale."}, {"date": "2023/06/15", "event": "User wore their mother\'s locket to their cousin\'s wedding."}, {"date": "2023/07/03", "event": "User found a single pearl earring while cleaning out their jewelry box."}]'),
                    ("Conversation date: 2023/05/29 (Mon) 14:48\nHuman user messages:\n\nuser: I'm thinking of participating in the International Market next month and I need to confirm the details. Can you tell me what's the process for registering as a vendor and what are the fees involved?\nuser: I'm interested in the International Market that's happening in our town next month. I'm not sure about the exact name, but I know it's a popular event that attracts a lot of visitors. I plan to sell my handmade crafts, like candles and soaps. By the way, I've been pretty busy with local markets and events lately - I attended the Farmers' Market at the town square just three weeks ago and had a great time.\nuser: I'll try contacting the local tourism board to see if they have any information about the International Market. Do you think they would also have information about other upcoming events in the area, like the Holiday Market in December?\nuser: I'll contact the local tourism board to get more information about the International Market and the Holiday Market. By the way, I've been meaning to reach out to Rachel, the jewelry maker I met at the Craft Fair, to collaborate on a future project. Do you think I could also ask the tourism board if they know of any upcoming craft fairs or artisan markets in the area?\nuser: I'll ask the tourism board about upcoming craft fairs and artisan markets, and also ask if they know of any resources that might be helpful for artisans like me. Do you think they would also know about any local classes or workshops that might help me improve my craft-making skills?\nuser: That's a great point about the tourism board's focus. I'll ask them about resources for artisans, but I'll also reach out to local community colleges, craft stores, and art organizations to find out about classes or workshops that can help me improve my craft-making skills. By the way, I've been meaning to restock my candle supplies, do you think the craft stores would also have information on local suppliers for materials like wax, essential oils, or fragrances?\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[{"date": "2023/05/08", "event": "Attended the Farmers\' Market at the town square."}]'),
                    ("Conversation date: 2023/05/20 (Sat) 20:47\nHuman user messages:\n\nuser: I'm looking for some recommendations on pet grooming tools. I've been using a Furminator brush on Luna, but I'm wondering if there are other products that could help with her shedding. By the way, I recently got her a new pet bed from Petco, and it was originally $40, but I'm really happy with the purchase!\nuser: I'm actually looking for recommendations for both Luna and Max. Max is a 5-year-old golden retriever, and Luna is a cat. I've been giving Luna Omega-3 supplements to help with her skin and coat health, so I'm open to trying out new products that can complement those supplements.\nuser: I'm thinking of getting a new leash for Max. Do you have any recommendations?\nuser: I was thinking of getting a hands-free leash from REI, maybe one that would allow me to run with Max if I want to.\nuser: I'm thinking of getting a new pet camera to keep an eye on Luna and Max when I'm not at home. I recently got a new pet bed for Luna, and I want to make sure she's using it properly. By the way, the original price of Luna's pet bed was $40.\nuser: Luna is loving her new pet bed! She's been sleeping in it every night, and it's so soft and plush. I'm glad I got it for her. By the way, I got it from Petco, and as I mentioned earlier, the original price was $40.\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[{"date": "2023/05/20", "event": "User got a new pet bed for Luna from Petco, originally priced at $40."}]'),
                    ("Conversation date: 2023/05/22 (Mon) 09:23\nHuman user messages:\n\nuser: I'm looking for some recommendations for similar bands to The Electric Storm, I just saw them live for the first time at the Music Festival at the Outdoors Pavilion and I'm hooked!\nuser: I'm positive it was The Electric Storm, I even got a t-shirt with their album artwork on it from the merchandise booth! They had a really high-energy performance and the crowd loved them. The opening act, Whiskey Wanderers, was great too. I'm not sure about specific songs, but their sound was kinda like a mix of classic rock and indie.\nuser: I was wondering if you can also recommend some local music venues in my area, since I've been to a bunch of concerts recently and I'm looking to support more local talent.\nuser: I've been to a few venues recently, actually. I went to the Rock on the River concert series at the Riverfront Amphitheater, and I also attended a singer-songwriter night at the Coffee House on Main Street, which was really intimate and cool. I've also been to the Arena for a Bruno Mars concert, and the Community Center for a benefit concert called Music for a Cause.\nuser: I was thinking of checking out the Underground Club, I've heard they have a great selection of indie and up-and-coming bands. Do you know if that's a good spot?\nuser: I'm still looking for some recommendations for similar bands to The Electric Storm. You mentioned some bands earlier, but I was wondering if you knew of any other bands that have a similar sound to theirs - a mix of classic rock and indie. Do you know of any other bands that might fit the bill?\n\nPersonal facts about the user with dates (a list of dicts in json format; do not generate anything else):", '[{"date": "2023/05/22", "event": "Saw The Electric Storm live for the first time at the Music Festival at the Outdoors Pavilion."}, {"date": "2023/05/22", "event": "Bought a t-shirt with The Electric Storm\'s album artwork from the merchandise booth."}, {"date": "2023/05/22", "event": "Saw Whiskey Wanderers as the opening act for The Electric Storm."}, {"date": "2023/05/22", "event": "Went to the Rock on the River concert series at the Riverfront Amphitheater."}, {"date": "2023/05/22", "event": "Attended a singer-songwriter night at the Coffee House on Main Street."}, {"date": "2023/05/22", "event": "Went to a Bruno Mars concert at the Arena."}, {"date": "2023/05/22", "event": "Attended a benefit concert called Music for a Cause at the Community Center."}]'),
                ]
                expansion = extract_session_userfact(cur_date, sess_entry, model_name, examples=examples)
            cur_expansions.append(expansion)
        entry['timestamped_facts'] = cur_expansions
        data.append(entry)
    
    json.dump(data, open(cache_file, 'w'))
