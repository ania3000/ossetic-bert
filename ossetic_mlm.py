from datasets import load_dataset, Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from huggingface_hub import login
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--hf_token', type=str, required=True)

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--tokenizer_name', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--hub_model_id', type=str, required=True)

parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument('--eval_steps', type=int, default = 200)
parser.add_argument('--save_steps', type=int, default = 200) 
parser.add_argument('--per_device_train_batch_size', type=int, default = 8)
parser.add_argument('--per_device_eval_batch_size', type=int, default = 8)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--max_steps", type=int)
group.add_argument("--num_train_epochs", type=int)

parser.add_argument('--sub_train', type=int, default=0)
parser.add_argument('--sub_test', type=int, default=0)

args = parser.parse_args()
token=args.hf_token


login(token=token)

model_name = args.model_name
tokenizer_name = args.tokenizer_name
dataset_name = args.dataset_name
hub_model_id = args.hub_model_id

model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
datasets = load_dataset(dataset_name, token=token)

sub_train = args.sub_train
sub_test = args.sub_test

train = datasets["train"]
test = datasets["test"]

if sub_train != 0:
    train = Dataset.from_dict(datasets["train"][:sub_train])
if sub_test != 0:
    test  = Dataset.from_dict(datasets["test"][:sub_test])

datasets = DatasetDict({"train": train, "test": test})

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

def group_sentences(examples, block_size=128):
    input_ids_list = examples["input_ids"]
    attention_mask_list = examples["attention_mask"]
    token_type_ids_list = examples.get("token_type_ids", None)

    grouped_input_ids = []
    grouped_attention_mask = []
    grouped_token_type_ids = [] if token_type_ids_list is not None else None

    current_ids, current_types, current_mask = [], [], []
    current_len = 0

    for sent_ids, sent_mask, sent_types in zip(
        input_ids_list,
        attention_mask_list,
        token_type_ids_list if token_type_ids_list is not None else input_ids_list
    ):
        sent_len = len(sent_ids)
        if sent_len > block_size:
            continue

        if current_len + sent_len <= block_size:
            current_ids.extend(sent_ids)
            current_mask.extend(sent_mask)
            if token_type_ids_list is not None:
                current_types.extend(sent_types)
            current_len += sent_len

        else:
            grouped_input_ids.append(current_ids)
            grouped_attention_mask.append(current_mask)
            if token_type_ids_list is not None:
                grouped_token_type_ids.append(current_types)

            current_ids = sent_ids.copy()
            current_mask = sent_mask.copy()
            current_types = sent_types.copy() if token_type_ids_list is not None else []
            current_len = sent_len

    result =  {
        "input_ids": grouped_input_ids,
        "attention_mask": grouped_attention_mask
    }

    if token_type_ids_list is not None:
        result["token_type_ids"] = grouped_token_type_ids

    return result

lm_datasets = tokenized_datasets.map(
    group_sentences,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

training_args_kwargs = dict(
    per_device_train_batch_size = args.per_device_train_batch_size,
    per_device_eval_batch_size = args.per_device_eval_batch_size,
    save_strategy="steps",
    eval_strategy="steps",
    logging_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    logging_steps = args.eval_steps,
    disable_tqdm=False,
    save_only_model=True,
    load_best_model_at_end=False,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    report_to="none",
    hub_model_id=args.hub_model_id,
    output_dir=args.output_dir
)

if args.max_steps is not None:
    training_args_kwargs["max_steps"] = args.max_steps
elif args.num_train_epochs is not None:
    training_args_kwargs["num_train_epochs"] = args.num_train_epochs

training_args = TrainingArguments(**training_args_kwargs)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
trainer.push_to_hub()