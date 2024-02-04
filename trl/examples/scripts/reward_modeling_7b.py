# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import random
import numpy as np
import torch
import wandb
from transformers import SchedulerType
import os
from huggingface_hub import login
import os
from peft import PeftModel, PeftConfig
os.environ['HUGGINGFACE_TOKEN'] = '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'
os.environ["WANDB_API_KEY"] = "082387cb19b12de064583cf2daae92afa1da455b"
login(token = "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")
# wandb.init()

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

from dataclasses import dataclass, field
from typing import Optional

import tyro
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from trl import RewardConfig, RewardTrainer, is_xpu_available


tqdm.pandas()
sys.path.append('/scratch/as14661/jup_env/rlhf_trojan_competition')
from src.models import RewardModel

@dataclass
class ScriptArguments:
    model_name: str = "None" #1. updated the model 
    """the model name"""
    dataset_name: str = "/scratch/as14661/medhalt/generations/medhalt_train_test" #4. Updated the data path
    """the dataset name"""
    dataset_text_field: str = "text"
    """the text field of the dataset"""
    eval_split: str = "none"
    """the dataset split to evaluate on; default to 'none' (no evaluation)"""
    load_in_8bit: bool = False
    """load the model in 8 bits precision"""
    load_in_4bit: bool = False
    """load the model in 4 bits precision"""
    trust_remote_code: bool = True
    """Enable `trust_remote_code`"""
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            per_device_train_batch_size=4, #7. reduced from 64 to 2
            num_train_epochs=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1e-6,
            report_to="wandb",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=1, # 3. Updated from 500 to 5
            evaluation_strategy="no",
#             eval_steps = 16,
            max_length=512,
            output_dir = '/scratch/as14661/trl/model_checkpoints/',
            save_steps = 10,
            save_strategy = "steps",
            lr_scheduler_type=SchedulerType.LINEAR,  # Specify the scheduler type
            warmup_steps=0, 
#             num_cycles = reduce to 10% of initial learning rate
            
        )
    )
    use_peft: bool = True
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=8,
            bias="none",
            task_type="SEQ_CLS",
            lora_dropout = 0.1,
        ),
    )


args = tyro.cli(ScriptArguments)
args.reward_config.evaluation_strategy = "steps" if args.eval_split != "none" else "no"


# Step 1: Load the model
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif args.load_in_8bit or args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
else:
    device_map = None
    quantization_config = None
    
##### Loading the model
base_model_name_or_path = "/scratch/as14661/jup_env/llama-2-7b-chat-hf"
peft_model_id = "vincentmin/llama-2-7b-reward-oasst1"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name_or_path,
    num_labels=1,
    load_in_4bit=True,
    torch_dtype=torch.float16,
)
peft_model = PeftModel.from_pretrained(model, peft_model_id)
merged_model = peft_model.merge_and_unload()

# Step 2: Load the dataset and pre-process it
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # 6. Added a padding token - update/change this
tokenizer.pad_token_id = tokenizer.eos_token_id


train_dataset = load_from_disk(args.dataset_name) #2. update the dataset
train_dataset = train_dataset["train"] #2b. updated this

merged_model.config.pad_token_id = model.config.eos_token_id # added padding token to the model 


# # modulating which layers to freeze
# last_mlp_layer = model.model.layers[-1].mlp
# for name, param in model.named_parameters(): # freezing all but the last layer
#      if not name.startswith("score"): # choose whatever you like here
#         param.requires_grad = False
        
# # unfreezing RMS norm for the last layer
# model.model.norm.requires_grad_(True)
        
"""unfreezing some of the mlp layers"""
## gate decides how much from the attention layer will flow to the mlp
# for param in last_mlp_layer.gate_proj.parameters(): 
#     param.requires_grad = True

## this is first upsampled
# for param in last_mlp_layer.up_proj.parameters():
#     param.requires_grad = True

## then downsampled
# for param in last_mlp_layer.down_proj.parameters():
#     param.requires_grad = True
    
# for param in model.model.layers[-1].input_layernorm.parameters():
#     param.requires_grad = True

# for param in model.model.layers[-1].post_attention_layernorm.parameters():
#     param.requires_grad = True


# # train only the final layer
# for name, param in model.named_parameters():
#     print (param.requires_grad)


# updating the number of warmup steps
num_train_samples = len(train_dataset)
total_train_batch_size = args.reward_config.per_device_train_batch_size * args.reward_config.gradient_accumulation_steps
total_steps = (num_train_samples / total_train_batch_size) * args.reward_config.num_train_epochs
warmup_steps = int(total_steps * 0.1)
args.reward_config.warmup_steps = warmup_steps

# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


# Preprocess the dataset and filter out examples that are longer than args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
    and len(x["input_ids_rejected"]) <= args.reward_config.max_length
)

if args.eval_split == "none":
    eval_dataset = None
else:
    eval_dataset = load_from_disk(args.dataset_name)
    eval_dataset = eval_dataset[args.eval_split] #5. updated how we are getting the eval_dataset

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
        and len(x["input_ids_rejected"]) <= args.reward_config.max_length
    )


# Step 4: Define the LoraConfig
if args.use_peft:
    peft_config = args.peft_config
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = RewardTrainer(
    model=merged_model,
    tokenizer=tokenizer,
    args=args.reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

print("Current working directory is", os.getcwd())

try:
    if not os.path.exists('/scratch/as14661/trl/model_checkpoints/'):
        os.makedirs('/scratch/as14661/trl/model_checkpoints/')
    trainer.train()
except KeyboardInterrupt:
    print ("KeyboardInterrupt")
