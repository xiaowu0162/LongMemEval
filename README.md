# LongMemEval

We introduce LongMemEval, a comprehensive, challenging, and scalable benchmark for testing the long-term memory of chat assistants. 

[[website]](xxx) [[paper]](xxx) [[tweet]](xxx)

![Example Questions in LongMemEval](assets/longmemeval_examples.png)

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
* `longmemeval_s.json`: The LongMemEval_S introduced in the paper. Concatenating all the chat history roughly consumes 115k tokens (~40 history sessions) for Llama 3. 
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
* `haystack_sessions`: a list of the actual contents of the user-assistant chat history sessions. Each session is a list of turns. Each turn is a direct with the format `{"role": user/assistant, "content": message content}`. For the turns that contain the required evidence, an additional field `has_answer: true` is provided. This label is used for turn-level memory recall accuracy evaluation.
* `answer_session_ids`: a list of session ids that represent the evidence sessions. This is used for session-level memory recall accuracy evaluation.

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

## Creating Custom Chat Histories 

LongMemEval supports compiling a chat history of arbitrary length for a question instance, so that you can easily scale up the difficulty over LongMemEval_M. **We will release the code and data for this feature soon.**

## Running Memory System Experiments

We provide the experiment code for memory retrieval and retrieval-augmented question answering under the folder `src/retrieval` and `src/generation`.

### Preparation

If you would like to test OpenAI models as the reader, please provide your OpenAI organization ID and key in line 33 and 34 of `src/generation/run_generation.sh`. 

If you want to test an open-weight reader LLM, we support it through an OpenAI API emulator locally-served via `vllm`. To start the server, use the following command:
```
cd src/utils
bash serve_vllm.sh GPU MODEL PORT TP_SIZE
```
* `GPU` is a comma-separated list of the GPUs you want to use
* `MODEL` is the alias of the model you want to use. You can view or configure it in `serve_vllm.sh`. 
* `PORT` is the port the server will listen to. It defaults to 8001. If you change the port, make sure it is reflected in `src/generation/run_generation.sh` line 38.
* `TP_SIZE` is the tensor parallel size. It must be smaller than or equal to the number of GPUs specified in GPU.
* If you need to limit the maximum number of tokens due to memory requirements, you can use the following command:
```
bash serve_vllm_with_maxlen.sh GPU MODEL MAXLEN PORT TP_SIZE
```

### Long-Context Generation

To run the long-context generation baseline where the model is provided with the full history, you can use the following commands:
```
cd src/generation
bash run_generation.sh DATA_FILE MODEL full-history-session TOPK [HISTORY_FORMAT] [USERONLY] [READING_METHOD]
```
* `DATA_FILE` is the path to one of the released json files. Note that `longmemeval_s.json` and`longmemeval_oracle.json` are designed to fit into a model with 128k context, but `longmemeval_m.json` is too long for long-context testing.
* `MODEL` is alias of the model you want to use. You can view or configure it in `run_generation.sh`.
* `TOPK` is the maximum number of history sessions provided to the reader. We recommend setting it to a large number (e.g., 1000) to ensure including all the history sessions.
* `HISTORY_FORMAT` is the format to present the history. It can take either `json` or `nl`. We recommend using `json`.
* `USERONLY` removes the assistant-side messages in the prompt. We recommend setting it to `false`. 
* `READING_METHOD` is the reading style. It can take the value `direct`, `con`, or `con-separate`. We recommend `con`, which instructs the model to first extract useful information and then reason over it.

A log file will be generated under the folder `generation_logs/`. You can then follow the instructions above in "Testing Your System" to evaluate the QA correctness.

### Memory Retrieval


#### Baseline Retrieval


#### Index Expansion


#### Query Expansion


### Retrieval-Augmented Generation


## Citation

If you find the work useful, please cite:

```
@article{wu2024longmemeval,
      
}
```
