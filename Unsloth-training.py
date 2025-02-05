from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

alpaca_prompt = """
### Instruction:
{}

### Input:
{}
You are the official marketing and communications assistant for employees at Massive Dynamic, a fictional multi-faceted conglomerate depicted in "Fringe", the American science fiction television series on the Fox television network. Reflect these brand attributes in tone and style:
- Innovating and cutting-edge
- Confident and assured
- Optimistic and visionary
- Sleek and modern
- Slightly mysterious, inviting curiosity.

All your responses should:
1. Reflect these brand attributes in tone and style.
2. Stay consistent with fictional facts about Massive Dynamic (e.g., founder, product lines) when asked.
3. Adapt the format and structure of your response to match the user’s query (e.g., press release, product description, commercial script).
4. Avoid real-world controversies or sensitive topics. Remember, the brand is fictional.

### Output:
{}
"""

EOS_TOKEN = tokenizer.eos_token     # Must add EOS_TOKEN or generation doesn't end.
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("CylinderS/MassDyn-Style-Guide-Data-300", split = "train")
split_dataset = dataset.train_test_split(test_size=0.1, seed=3407)

train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
val_dataset = val_dataset.map(formatting_prompts_func, batched = True,)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    do_train = True,    # Training won't happen without this.
    do_eval = True,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        evaluation_strategy = "steps", # Can be steps or epochs
        eval_steps = 5,
        report_to = "none",
    ),
)

trainer_stats = trainer.train()

# Some troubleshooting and debugging statements

# print("Trainer arguments:", trainer.args)
# print("Train dataset length:", len(train_dataset))
# print("Val dataset length:", len(val_dataset))
# print(train_dataset[200])
# print("EOS token is:", repr(tokenizer.eos_token))

model.save_pretrained("LLMs/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit")
tokenizer.save_pretrained("LLMs/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit")
# model.push_to_hub("CylinderS/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit", token = "hf_zzz") # Online saving
# tokenizer.push_to_hub("CylinderS/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit", token = "hf_zzz") # Online saving

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("LLMs/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit", tokenizer,)
if False: model.push_to_hub_gguf("CylinderS/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit", tokenizer, token = "hf_zzz")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("LLMs/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("CylinderS/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit", tokenizer, quantization_method = "f16", token = "hf_zzz")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("LLMs/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("CylinderS/MassDyn-StyleGuide-Llama3.1-8B-Instruct-4bit", tokenizer, quantization_method = "q4_k_m", token = "hf_zzz")

# Save to multiple GGUF options; faster than multiple above steps
if False:
    model.push_to_hub_gguf(
        "hf-username/hf-modelrepo",
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )