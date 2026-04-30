from gpt_structure import client
from prompt import SPEAKER_PROMPT, LISTENER_PROMPT
from typing import *
import time
import os
import random
import base64
import logging
from data import demo_object,Color,Shape,all_objects


LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_gpt_prompt.log")
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
def list_to_table(lst:List[str],obj:str)->str:
    if not lst:
        raise ValueError("列表不能为空")

    # 编号宽度（根据最大编号长度来自动对齐）
    index_width = len(str(len(lst)))
    # 内容宽度（最长的元素字符串）
    content_width = max(len(str(item)) for item in lst)

    # 构造表格行
    lines = []
    header = f"{'Serial Number'.ljust(index_width)} | {obj.ljust(content_width)}"
    divider = '-' * len(header)
    lines.append(header)
    lines.append(divider)

    for idx, item in enumerate(lst, 1):
        lines.append(f"{str(idx).ljust(index_width)} | {str(item).ljust(content_width)}")

    return '\n'.join(lines)
def speaker_generate(
    target_object:demo_object,
    learned_vocabulary:Dict[str, str],
    letters:List[str],
    max_word_len:int,
    model:str="gpt-4o"
)->str:
    learned_vocabulary = {str(k): v for k, v in learned_vocabulary.items()}
    logger.info("speaker_generate started: target=%s, vocab_size=%s", target_object, len(learned_vocabulary))
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }

    # Build image-text pairs from learned_vocabulary
    content = [{"type":"text", "text": "**LEARNED VOCABULARY**\n"}]

    # Add image-text pairs for each entry in learned_vocabulary
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    for key, value in learned_vocabulary.items():
        image_dir = os.path.join(demo_dir, "images", key)
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                chosen_img = random.choice(images)
                img_path = os.path.join(image_dir, chosen_img)
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
                content.append({
                    "type": "text",
                    "text": value
                })
    content.append({
                    "type": "text",
                    "text": f"**TARGET IMAGE**\n"
                })
    image_dir = os.path.join(demo_dir, "images", str(target_object))
    if os.path.exists(image_dir):
        images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            chosen_img = random.choice(images)
            img_path = os.path.join(image_dir, chosen_img)
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
    # Format the prompt with inputs
    prompt_inputs = [
        str(len(letters)),                          # !<INPUT 0>! - number of letters
        list_to_table(letters,"Letters"),           # !<INPUT 1>! - letters              # !<INPUT 3>! - vocab size
        str(max_word_len),                          # !<INPUT 2>! - max word length
    ]
    formatted_prompt = SPEAKER_PROMPT
    for idx, inp in enumerate(prompt_inputs):
        formatted_prompt = formatted_prompt.replace(f"!<INPUT {idx}>!", inp)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": formatted_prompt},
        ]
    }]

    # Insert image-text pairs after the first text element
    messages[0]["content"] = [messages[0]["content"][0]] + content

    time.sleep(0.5)
    response = client.chat.completions.create(
        model=gpt_param["model"],
        messages=messages,
        max_tokens=gpt_param["max_tokens"],
        top_p=gpt_param["top_p"],
        frequency_penalty=gpt_param["frequency_penalty"],
        presence_penalty=gpt_param["presence_penalty"],
        temperature=gpt_param["temperature"],
    )
    result = response.choices[0].message.content
    logger.info("speaker_generate output: %s \n", result)
    return result

def listener_select(
    word:str,
    target_object:demo_object,
    learned_vocabulary:Dict[str, str],
    letters:List[str],
    max_word_len:int,
    model:str="gpt-4o"
)->Tuple[str, Dict[str, demo_object]]:
    learned_vocabulary = {str(k): v for k, v in learned_vocabulary.items()}
    # Sample choices: 1 from target_object, 4 random from other objects, then shuffle
    other_objects = [obj for obj in all_objects if obj != target_object]
    random_choices = random.sample(other_objects, min(4, len(other_objects)))
    choices = [target_object] + random_choices
    random.shuffle(choices)

    logger.info("listener_select started: word=%s, target=%s, vocab_size=%s, shuffled_choices=%s",
                word, target_object, len(learned_vocabulary), choices)
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }

    # Build content: learned vocabulary + word + choice images
    content = [{"type":"text", "text": "**LEARNED VOCABULARY**\n"}]

    # Add image-text pairs for each entry in learned_vocabulary
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    for key, value in learned_vocabulary.items():
        image_dir = os.path.join(demo_dir, "images", key)
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                chosen_img = random.choice(images)
                img_path = os.path.join(image_dir, chosen_img)
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
                content.append({
                    "type": "text",
                    "text": value
                })

    # Add the word to guess
    content.append({"type":"text", "text": f"**WORD TO GUESS**: {word}\n"})
    content.append({"type":"text", "text": "**CHOICE IMAGES**\n"})

    # Add 5 choice images with labels A-E
    choice_labels = ["A", "B", "C", "D", "E"]
    choice_map = {choice_labels[i]: choices[i] for i in range(len(choices))}
    for i, obj in enumerate(choices):
        image_dir = os.path.join(demo_dir, "images", str(obj))
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                chosen_img = random.choice(images)
                img_path = os.path.join(image_dir, chosen_img)
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"[{choice_labels[i]}]"
                })

    # Format the prompt with inputs
    prompt_inputs = [
        str(len(letters)),                          # !<INPUT 0>! - number of letters
        list_to_table(letters,"Letters"),           # !<INPUT 1>! - letters
        str(max_word_len),                          # !<INPUT 2>! - max word length
    ]
    formatted_prompt = LISTENER_PROMPT
    for idx, inp in enumerate(prompt_inputs):
        formatted_prompt = formatted_prompt.replace(f"!<INPUT {idx}>!", inp)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": formatted_prompt},
        ]
    }]

    # Insert image-text pairs after the first text element
    messages[0]["content"] = [messages[0]["content"][0]] + content

    time.sleep(0.5)
    response = client.chat.completions.create(
        model=gpt_param["model"],
        messages=messages,
        max_tokens=gpt_param["max_tokens"],
        top_p=gpt_param["top_p"],
        frequency_penalty=gpt_param["frequency_penalty"],
        presence_penalty=gpt_param["presence_penalty"],
        temperature=gpt_param["temperature"],
    )
    result = response.choices[0].message.content
    logger.info("listener_select output: %s \n", result)
    return result, choice_map
