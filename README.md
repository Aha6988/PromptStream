# PromptStream
The official code for the paper "PromptStream: Self-Supervised News Story Discovery Using Topic-Aware Article Representations"
### Setup
1. Create a venv and install requirements.txt.
2. Edit the ''Class config'' to choose a dataset or edit the parameters.
3. Run with python prompt_stream.py.
### Datasets
1. The datasets used in the paper can be downloaded from https://github.com/cliveyn/SCStory
2. For custom dataset, change dataset="custom" and specify the path in CUSTOM_DATASET. The dataset needs to contain the columns [date(e.g.'2024-05-22'), title(str), body(str), cluster(int)].
### Outputs
The model prints the batch size and metrics every.
### Detailed configuration
#### Set the window size
The window size for the sliding window is set by the parameters q_size and slide_size.
#### Init training parameters
The parameters for training the prompt representations
#### Continual training parameters
sample_thr =
#### Try different prompt templates