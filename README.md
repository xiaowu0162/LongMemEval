# LongMemEval

We introduce LongMemEval, a comprehensive, challenging, and scalable benchmark for testing the long-term memory of chat assistants. 

[[website]](xxx) [[paper]](xxx) [[tweet]](xxx)


[Add a teaser figure here]


## Setup

### Data

The LongMemEval dataset is officially released [here](https://drive.google.com/file/d/1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80/view?usp=sharing). Please download and uncompress the data to the `data/` folder. 
```
mkdir -p data/
mv longmemeval_data.tar.gz data/ 
cd data ; tar -xzvf longmemeval_data.tar.gz ; cd ..
```

### Environment

We recommend using a conda environment for the project. You may follow the steps below to set up.

#### Evaluation only

If you only need to calculate the metrics on the outputs produced by your own system, you can install this minimal requirement set which allows you to run `src/evaluation/report_metrics.py`.

```
conda create -n longmemeval-lite python=3.9
conda activate longmemeval-lite
pip install -r requirements-lite.txt
```

#### Full support

If you also would like to run the memory systems introduced in the paper, please set up this environment instead. 

```
conda create -n longmemeval python=3.9
conda activate longmemeval
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-full.txt
```
We have tested this environment on a Linux machine with CUDA 12.1. If you use a different platform, you may need to modify the requirements.

## Dataset Format

Three files are included in the data package:
* `longmemeval_s.json`: The LongMemEval_S introduced in the paper. Concatenating all the chat history roughly consumes 115k tokens for Llama 3. 
* `longmemeval_m.json`: The LongMemEval_M introduced in the paper. Each chat history contains roughly 500 sessions. 
* `longmemeval_oracle.json`: LongMemEval with oracle retrieval. Only the evidence sessions are included in the history. 

Within each file, there are 500 evaluation instances, each of which contains the following fields:
* `question_id`: the unique id for each question.
* `question_type`: one of `single-session-user`, `single-session-assistant`, `single-session-preference`, `temporal-reasoning`, `knowledge-update`, and `multi-session`. If `question_id` ends with `_abs`, then the question is an `abstention` question. 
* `question`: the question content.
* `answer`: the expected answer from the model.
* `question_date`: the date of the question.
* `haystack_session_ids`: a list of the ids of the history sessions (sorted by their timestamp). 
* `haystack_dates`: a list of the timestamps of the history sessions. 
* `haystack_sessions`: a list of the actual contents of the user-assistant chat history sessions. Each session is a list of turns. Each turn is a direct with the format `{"role": xxx, "content": xxx}`. For the turns that contain the required evidence, an additional field `has_answer: true` is provided. This label is used for turn-level retrieval evaluation.
* `answer_session_ids`: a list of session ids that represent the evidence sessions. This is used for session-level retrieval evaluation.

## Testing Your System

To test on LongMemEval, you may directly feed the timestamped history to your own chat system, collect the output, and evaluate with the evaluation script we provide. To do so, save the outputs in a `jsonl` format with each line containing two fields: `question_id` and `hypothesis`. Then, you may run the evaluation script through the following command:

```
export OPENAI_ORGANIZATION=YOUR_ORGANIZATION
export OPENAI_API_KEY=YOUR_API_KEY
cd src/evaluation
python3 evaluate_qa.py gpt-4o your_hypothesis_file ../../data/longmemeval_oracle.json
```

Running this script will save the evaluation logs into a file called `[your_hypothesis_file].log`. In this file, each line will contain a new field called `autoeval_label`. While `evaluate_qa.py` already reports the averaged scores, you can also aggregate the scores from the log using the following command:

```
(assuming you are in the src/evaluation folder)
python3 print_qa_metrics.py gpt-4o your_hypothesis_file ../../data/longmemeval_oracle.json
```

## Reproducing Memory System Results

#### Long-Context Generation

#### Memory Retrieval

#### Retrieval-Augmented Generation


## Creating Custom Chat Histories 


## Citation

If you find the work useful, please cite:

```
@article{wu2024longmemeval,
      
}
```
