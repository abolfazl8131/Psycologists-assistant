import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import mlflow
import os

def fine_tune_llm(max_seq = 2048,load_in_4bit=True,warmup_steps=3,max_steps=10):

    if warmup_steps >= max_steps:
        return False

    ds = load_dataset("heliosbrahma/mental_health_chatbot_dataset", split='train')

    if os.path.isdir("psyco_assistant"):
        print('h')
        model,tokenizer = FastLanguageModel.from_pretrained(model_name = "./psyco_assistant")

    print('hi')
    model,tokenizer = FastLanguageModel.from_pretrained(model_name = "unsloth/tinyllama-bnb-4bit",
                                                        max_seq_length = max_seq, 
                                                        dtype=None, 
                                                        load_in_4bit=load_in_4bit)



    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    mlflow.set_experiment("Psyco assistant fine-tuning process")
    mlflow.set_tracking_uri(str(os.environ.get('MLFLOW_TRACKING_URI')))        
   
    trainer = SFTTrainer(
        model = model,
        train_dataset = ds,
        dataset_text_field = "text",
        max_seq_length = max_seq,
        tokenizer = tokenizer,
        args = TrainingArguments(
            
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = warmup_steps,
            max_steps = max_steps,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "unsloth-test",
            optim = "adamw_8bit",
            seed = 3407,
        ),
    )
    

    trainer.train()

    model.save_pretrained('psyco_assistant')
    tokenizer.save_pretrained('psyco_assistant')
    


if __name__ == '__main__':
    
    fine_tune_llm()