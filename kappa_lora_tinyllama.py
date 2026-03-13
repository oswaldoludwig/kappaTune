# kappa_lora_tinyllama.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import TaskType
from kappaTune.hf_integration import get_kappatune_lora_model
from transformers import DataCollatorForLanguageModeling

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# KappaTune-LoRA in ONE line!
model = get_kappatune_lora_model(
    model,
    num_modules_to_adapt=20,   # original example budget
    lora_rank=16,
    task_type=TaskType.CAUSAL_LM
)
model.print_trainable_parameters()  # will show only ~0.1-0.5% trainable

# Dataset + collator (standard)
dataset = load_dataset("ag_news", split="train[:5%]")  # or your data

# 1. Define the preprocessing function
def preprocess_function(examples):
    # We concatenate the news 'text' field.
    # For Causal LM, we usually just train on the text itself.
    return tokenizer(examples["text"], truncation=True, max_length=512)

# 2. Tokenize the dataset
# 'remove_columns' is important so the Trainer doesn't try to pass strings to the model
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 3. Data Collator
# This handles padding and creates 'labels' (clones of input_ids) automatically
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="kappa_lora_tinyllama",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    push_to_hub=True,   # ← auto-upload adapter!
    hub_model_id="yourusername/kappatune-lora-tinyllama-agnews",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()
trainer.push_to_hub()  # adapter + config saved automatically
