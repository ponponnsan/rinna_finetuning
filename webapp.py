import os
import datetime
import gradio as gr
from gradio.mix import Parallel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig


model_path = "rinna/japanese-gpt-neox-3.6b"
# cyberagent/open-calm-7b
base_llm = AutoModelForCausalLM.from_pretrained(model_path)
# , device_map="auto"
# torch_dtype=torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = PeftModel.from_pretrained(base_llm, 'lora-rinna-3.6b-sakura_dataset')
# torch_dtype=torch.float16
# model1.half()


# プロンプトテンプレートの準備

PROMPT_DICT = {
    "prompt_no_input": (
        "以下は、タスクを説明する指示です。"
        "要求を適切に満たす応答を書きなさい。\n\n"
        "### 入力:\n{input}\n\n### 応答:"
    )
}

def map(input):
    m = {
    'input': input,
    }
    return m
    


# テキスト生成関数の定義
def generate(input=None,maxTokens=256):
    # 推論
    print(input)
    prompt = PROMPT_DICT['prompt_no_input'].format_map(map(input))
    print(prompt)
    inputs = tokenizer(
        prompt,
        return_tensors='pt'
    ).to(model.device)
    print('WWWIDIDJWIFJWIFJWIF')
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)

    output = output.replace('<NL>', '\n')
    output = output.split('\n\n')[-1].split(':')[-1]
    print("OOOOOOOOOOOUUUUUUUUUUUUUTTTTTTTTT22222222222", output)

    return output




# def chat(model, instruction, temp, conversation=''):
    
#     # 温度パラメータの入力制限
#     if temp < 0.1:
#         temp = 0.1
#     elif temp > 1:
#         temp = 1
    
#     input_prompt = {
#         'instruction': instruction,
#         'input': conversation
#     }
    
#     prompt = ''
#     if input_prompt['input'] == '':
    
#         prompt = PROMPT_DICT['prompt_no_input'].format_map(input_prompt)
#     else:
#         prompt = PROMPT_DICT['prompt_input'].format_map(input_prompt)

    
#     inputs = tokenizer(
#         prompt,
#         return_tensors='pt'
#     ).to(model.device)
    
#     with torch.no_grad():
#         tokens = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             do_sample=True,
#             temperature=temp,
#             top_p=0.9,
#             repetition_penalty=1.05,
#             pad_token_id=tokenizer.pad_token_id,
#         )

#     output = tokenizer.decode(tokens[0], skip_special_tokens=True)
#     output = output.split('\n\n')[-1].split(':')[-1]
    
#     return output

def fn1(input):
    return generate(input,maxTokens=256)



examples = [
    ["日本の観光名所を3つ挙げて。"],
    ["データサイエンティストに必要なスキルを5つ挙げて。"],
    ["3+2は？"]
]

demo1= gr.Interface(
    fn1,
    [
        gr.inputs.Textbox(lines=5, label="入力テキスト"),
        # gr.inputs.Number(default=0.3, label='Temperature(0.1 ≦ temp ≦ 1 の範囲で入力してください\n小さい値ほど指示に忠実、大きい値ほど多様な表現を出力します。)'),
    ],
    gr.outputs.Textbox(),
)

demo = gr.Parallel(
    demo1,
    examples=examples,
    title="rinnnaちゃん",
    # CyberAgent Open-CALM-7b-lora-instruction-tuning
    description="""
        LoRAのチューニング方法
        """,)

# share=Trueで外部に公開される。
demo.launch(server_name="0.0.0.0", share=True)



# def greet(name):
#     return "Hello " + name + "!!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# # servernameをしてしてあげないと、外部から見えない。
# demo.launch(server_name="0.0.0.0", share=True)