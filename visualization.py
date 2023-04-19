# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.8.8 ('base')
#     language: python
#     name: python3
# ---

# + id="pujJkh_q8qTi"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, DataCollatorWithPadding

from transformers import TrainingArguments, Trainer
from datasets import Dataset
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

# + colab={"base_uri": "https://localhost:8080/"} id="iOkFdQoT8w8X" outputId="aaf80ccc-4f67-4342-f2f1-ba46f3cb2d19"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps:0")
# -

ds_train = Dataset.from_csv('train_twitter.csv')
ds_val = Dataset.from_csv('validate_twitter.csv')
ds_test = Dataset.from_csv('test_twitter.csv')
ds = {"train": ds_train, "validation": ds_val, "test": ds_test}


id2label = {0: "United States", 1: "United Kingdom", 2: "Canada", 3: "Australia", 4: "India", 5: "Nigeria"}
label2id = {"United States": 0, "United Kingdom": 1, "Canada": 2, "Australia": 3, "India": 4, "Nigeria": 5}

# + colab={"base_uri": "https://localhost:8080/"} id="ZptRohBG9CCK" outputId="f70a08b8-275d-400e-b85d-b63b06269f03"
model_path = 'my_awesome_model'

model = AutoModelForSequenceClassification.from_pretrained("Twitter/twhin-bert-base", num_labels=6, id2label=id2label, label2id=label2id)
model = model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')

def preprocess_function(examples):
    label = examples["country"] 
    examples = tokenizer(examples["tweet_text"], truncation=True, padding="max_length", max_length=256, return_tensors='pt')
    for key in examples:
        examples[key] = examples[key].squeeze(0)
    examples["label"] = label
    return examples

for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=['user_id', 'tweet_id', 'tweet_text', 'country'])
    ds[split].set_format('pt')


# +
import evaluate

accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels), "f1":f1_metric.compute(predictions=predictions, references=labels, average="weighted")}


# -

import torch
class TwitterTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # print ("inputs: ", inputs)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


# +
from transformers import TrainingArguments
from transformers import Trainer

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = TwitterTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics
)

# -

device


# +
def forward_func(inputs, position = 0):
    """
        Wrapper around prediction method of pipeline
    """
    pred = model(inputs, attention_mask=torch.ones_like(inputs).to(device))
    return pred[position]
    
def visualize(inputs: list, attributes: list):
    """
        Visualization method.
        Takes list of inputs and correspondent attributs for them to visualize in a barplot
    """
    attr_sum = attributes.sum(-1) 
    
    attr = attr_sum / torch.norm(attr_sum)
    
    a = pd.Series(attr.numpy()[0], 
                        index = tokenizer.convert_ids_to_tokens(inputs.detach().numpy()[0]))
    
    plt.show(a.plot.barh(figsize=(10,20)))
                    
def explain(text: str):
    """
        Main entry method. Passes text through series of transformations and through the model. 
        Calls visualization method.
    """
    prediction = trainer.predict(text)
    inputs = generate_inputs(text)
    baseline = generate_baseline(sequence_len = inputs.shape[1])
    
    lig = LayerIntegratedGradients(forward_func, getattr(model, 'Twitter/twhin-bert-base').embeddings)
    
    attributes, delta = lig.attribute(inputs=inputs,
                                baselines=baseline,
                                target = model.config.label2id[prediction[0]['label']], 
                                return_convergence_delta = True)
    
    visualize(inputs, attributes, prediction)
    
def generate_inputs(text: str):
    """
        Convenience method for generation of input ids as list of torch tensors
    """
    return tokenizer.encode(text, truncation=True, max_length=256, return_tensors='pt').to(device)

def generate_baseline(sequence_len: int):
    """
        Convenience method for generation of baseline vector as list of torch tensors
    """        
    return torch.tensor([tokenizer.cls_token_id] + [tokenizer.pad_token_id] * (sequence_len - 2) + [tokenizer.sep_token_id], device = device).unsqueeze(0)



# -

factory = iter(ds['test'])
text = iter(ds_test)

example = next(factory)
example_text = next(text)

example_text['tweet_text']

prediction = trainer.predict([example])


inputs = generate_inputs(example_text['tweet_text'])
baseline = generate_baseline(sequence_len = inputs.shape[1])
lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

attributes, delta = lig.attribute(inputs=inputs,
                            baselines=baseline,
                            target = model.config.label2id[model.config.id2label[prediction.label_ids[0]]], 
                            return_convergence_delta = True)

visualize(inputs, attributes)

# +

inputs = generate_inputs(example_text['tweet_text'])
baseline = generate_baseline(sequence_len = inputs.shape[1])

lig = LayerIntegratedGradients(forward_func, getattr(model, 'Twitter/twhin-bert-base').embeddings)

attributes, delta = lig.attribute(inputs=inputs,
                            baselines=baseline,
                            target = model.config.label2id[model.config.id2label[prediction.label_ids[0]]], 
                            return_convergence_delta = True)

visualize(inputs, attributes, prediction)
# -

rec = viz.VisualizationDataRecord(attr, 
                            .9,
                            model.config.label2id[model.config.id2label[prediction.label_ids[0]]],
                            0,
                            id2label[0],
                            attributes.sum(),
                            tokenizer.convert_ids_to_tokens(inputs.detach().numpy()[0]),
                            delta)


attr = attributes.sum(dim=2).squeeze(0)
attr = attr / torch.norm(attr)
attr = attr.cpu().detach().numpy()


tokenizer.convert_ids_to_tokens(inputs.detach().numpy()[0])

viz.visualize_text([rec])

# +

ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence


# Below we define a set of helper function for constructing references / baselines for word tokens, token types and position ids. We also provide separate helper functions that allow to construct attention masks and bert embeddings both for input and reference.

# In[7]:


def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] +         [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_whole_bert_embeddings(input_ids, ref_input_ids,                                     token_type_ids=None, ref_token_type_ids=None,                                     position_ids=None, ref_position_ids=None):
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids)
    
    return input_embeddings, ref_input_embeddings


# Let's define the `question - text` pair that we'd like to use as an input for our Bert model and interpret what the model was forcusing on when predicting an answer to the question from given input text 

# In[8]:


question, text = "What is important to us?", "It is important to us to include, empower and support humans of all kinds."


# Let's numericalize the question, the input text and generate corresponding baselines / references for all three sub-embeddings (word, token type and position embeddings) types using our helper functions defined above.

# In[9]:


input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
attention_mask = construct_attention_mask(input_ids)

# +



def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


# In[14]:


attributions_start_sum = summarize_attributions(attributions_start)
attributions_end_sum = summarize_attributions(attributions_end)


# In[15]:


# storing couple samples in an array for visualization purposes
start_position_vis = viz.VisualizationDataRecord(
                        attributions_start_sum,
                        torch.max(torch.softmax(start_scores[0], dim=0)),
                        torch.argmax(start_scores),
                        torch.argmax(start_scores),
                        str(ground_truth_start_ind),
                        attributions_start_sum.sum(),       
                        all_tokens,
                        delta_start)

end_position_vis = viz.VisualizationDataRecord(
                        attributions_end_sum,
                        torch.max(torch.softmax(end_scores[0], dim=0)),
                        torch.argmax(end_scores),
                        torch.argmax(end_scores),
                        str(ground_truth_end_ind),
                        attributions_end_sum.sum(),       
                        all_tokens,
                        delta_end)

print('\033[1m', 'Visualizations For Start Position', '\033[0m')
viz.visualize_text([start_position_vis])

print('\033[1m', 'Visualizations For End Position', '\033[0m')
viz.visualize_text([end_position_vis])
