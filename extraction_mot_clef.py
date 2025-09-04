import transformers
import torch
import re



model_id = "meta-llama/Llama-3.2-1B-Instruct"
#model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def faire_prompt_dialogue(dialogue):
    amorce = """Your task is to detect if the user is looking for a specific item in a picture.
If the user is looking for a specific item, you must name the item, using only one word. If the user is not looking for anything, just say "nothing".
"""
    prompt0 = {"role": "system", "content": amorce }
    dialogue = re.sub("\r\n", "\n", dialogue)
    prompt = [{k:v for k,v in prompt0.items()}]
    prompt.append({"role": "user", "content": "Is there a clock in the picture?"})
    prompt.append({"role": "assistant", "content": "clock"})
    prompt.append({"role": "user", "content": "Can you see a truck?"})
    prompt.append({"role": "assistant", "content": "truck"})
    prompt.append({"role": "user", "content": dialogue})

    return prompt

class ExtracteurMotClef:
    def __init__(self):
            self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            token = tokenHF,
            model_kwargs={"torch_dtype": torch.bfloat16},
            #device_map="auto", #Utilisation du module "Accelerate"
            device = "cuda", # GPU0
        )
        
    def traiter_dialogue(self, dlg):
        it_prompt = faire_prompt_dialogue(dlg)
        outputs = self.pipeline(
                it_prompt,
                max_new_tokens=16,
                #pad_token_id=pipeline.model.config.eos_token_id[0],  # eos_token_id est une liste le premier correspond à <|end_of_text|>
                do_sample=True,
                #num_beams=5,
                #continue_final_message = continue_final_message
            )
          resu = outputs[-1]["generated_text"][-1]["content"]
          return resu