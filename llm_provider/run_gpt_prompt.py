from llm_provider.gpt_structure import *
import asyncio
import logging
from model import image,all_images
from typing import List, Dict
import random
from llm_provider.prompt import SPEAKER_GENERATION, LISTENER_SELECTION,SPEAKER_RETRIEVAL,LISTENER_RETRIEVAL
default_logger = logging.getLogger(__name__)
default_logger.addHandler(logging.NullHandler())


def _get_logger(run_logger=None):
    return run_logger or default_logger


def _log_context(player_id=None, phase=None, round_num=None, comm_round=None)->str:
    return (
        f"player_id={player_id} "
        f"phase={phase} "
        f"round={round_num} "
        f"comm_round={comm_round}"
    )


def deal_json_format(text):
    text=text.replace("**EXPECTED FORMAT:**","")
    text=text.replace('json',"")
    text=text.replace('`','')
    text=text.strip()
    return text

def list_to_table(lst,obj):
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

def dict_list_to_str(dict_list):
    
    if not dict_list:
        return ""

    if not all(isinstance(d, dict) for d in dict_list):
        raise ValueError("All elements in the list must be dictionaries. ")

    return '\n'.join(str(d) for d in dict_list)

def split_cv_blocks(word: str)->List[str]:
    blocks=word.split("-")
    return blocks

def speaker_generate(
    target_object:image,
    learned_vocabulary:Dict[str, image],
    letters:List[str],
    max_word_len:int,
    model:str="gpt-4o",
    logger=None,
    player_id=None,
    phase=None,
    round_num=None,
    comm_round=None,
)->str:
    logger=_get_logger(logger)
    log_context=_log_context(player_id, phase, round_num, comm_round)
    logger.info(
        "speaker_generate started: %s target=%s vocab_size=%s",
        log_context,
        target_object,
        len(learned_vocabulary),
    )
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
    for word, im in learned_vocabulary.items():
        if im.path.exists():
            with open(im.path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
            content.append({
                "type": "text",
                "text": word
            })
    content.append({
                    "type": "text",
                    "text": f"**TARGET IMAGE**\n"
                })
    if target_object.path.exists():
        with open(target_object.path, "rb") as f:
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
    formatted_prompt = SPEAKER_GENERATION
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

    def __func_clean_up(gpt_response,
                        letters_list=letters):
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            word=gpt_response.get("word","")
            if (not word) or (not isinstance(word, str)):
                return False

            blocks=split_cv_blocks(word)
            for letter in blocks:
                if letter not in letters_list:
                    return False
            if len(blocks)>=1 and len(blocks)<=max_word_len:
                # gpt_response["word"]=word
                return gpt_response
            else:
                return False
                    
        except:
            return False

    cleaned_result = None
    result = ""
    attempts=5
    for attempt in range(attempts):
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
        cleaned_result = __func_clean_up(result)
        logger.info(
            "speaker_generate attempt: %s attempt=%s/%s valid=%s output=%s",
            log_context,
            attempt + 1,
            attempts,
            bool(cleaned_result),
            result,
        )
        if cleaned_result:
            break
    if not cleaned_result:
        logger.error("speaker_generate failed validation: %s output=%s", log_context, result)
        raise ValueError(f"speaker_generate returned invalid output after {attempts} attempts:\n{result}")
    logger.info("speaker_generate output: %s output=%s", log_context, result)
    return json.dumps(cleaned_result, ensure_ascii=False)

def listener_select(
    word:str,
    target_object:image,
    learned_vocabulary:Dict[str, image],
    letters:List[str],
    max_word_len:int,
    model:str="gpt-4o",
    choices:List[image]|None=None,
    logger=None,
    player_id=None,
    phase=None,
    round_num=None,
    comm_round=None,
)->str:
    logger=_get_logger(logger)
    log_context=_log_context(player_id, phase, round_num, comm_round)
    if choices is None:
        other_objects = [obj for obj in all_images if obj != target_object]
        random_choices = random.sample(other_objects, min(4, len(other_objects)))
        choices = [target_object] + random_choices
        random.shuffle(choices)
    else:
        choices = list(choices)
        if target_object not in choices:
            choices.append(target_object)

    logger.info(
        "listener_select started: %s word=%s target=%s vocab_size=%s shuffled_choices=%s",
        log_context,
        word,
        target_object,
        len(learned_vocabulary),
        choices,
    )
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
    for vocab_word, im in learned_vocabulary.items():
        if im.path.exists():
            with open(im.path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
            content.append({
                "type": "text",
                "text": vocab_word
            })

    # Add the word to guess
    content.append({"type":"text", "text": f"**WORD TO GUESS**: {word}\n"})
    content.append({"type":"text", "text": "**CHOICE IMAGES**\n"})

    # Add 5 choice images with labels A-E
    choice_labels = ["A", "B", "C", "D", "E"]
    displayed_choice_labels = []
    choice_map = {}
    for i, obj in enumerate(choices):
        if obj.path.exists():
            with open(obj.path, "rb") as f:
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
            displayed_choice_labels.append(choice_labels[i])
            choice_map[choice_labels[i]]=obj

    # Format the prompt with inputs
    prompt_inputs = [
        str(len(letters)),                          # !<INPUT 0>! - number of letters
        list_to_table(letters,"Letters"),           # !<INPUT 1>! - letters
        str(max_word_len),                          # !<INPUT 2>! - max word length
    ]
    formatted_prompt = LISTENER_SELECTION
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

    def __func_clean_up(gpt_response):
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            option=gpt_response.get("option","")
            if not option or not isinstance(option, str):
                return False
            option=option.strip().upper()
            if option in displayed_choice_labels:
                gpt_response["option"]=option
                gpt_response["selected_image_num"]=choice_map[option].num
                gpt_response["target_option"]=next(
                    label for label, obj in choice_map.items() if obj == target_object
                )
                return gpt_response  
            else:
                return False       
        except:
            return False

    cleaned_result = None
    result = ""
    attempts=5
    for attempt in range(attempts):
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
        cleaned_result = __func_clean_up(result)
        logger.info(
            "listener_select attempt: %s attempt=%s/%s valid=%s output=%s",
            log_context,
            attempt + 1,
            attempts,
            bool(cleaned_result),
            result,
        )
        if cleaned_result:
            break
    if not cleaned_result:
        logger.error("listener_select failed validation: %s output=%s", log_context, result)
        raise ValueError(f"listener_select returned invalid output after {attempts} attempts:\n{result}")
    logger.info("listener_select output: %s output=%s", log_context, result)
    return json.dumps(cleaned_result, ensure_ascii=False)

def speaker_retrieval(
                    target_image:image,
                    all_images:List[image],
                    model:str="gpt-4o",
                    logger=None,
                    player_id=None,
                    phase=None,
                    round_num=None,
                    comm_round=None)->str:

    logger=_get_logger(logger)
    log_context=_log_context(player_id, phase, round_num, comm_round)
    logger.info(
        "speaker_retrieval started: %s target=%s all_images_count=%s",
        log_context,
        target_image,
        len(all_images),
    )
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }

    # Build content: ALL IMAGES with serial numbers + GIVEN IMAGE
    content = [{"type":"text", "text": "**ALL IMAGES**\n"}]

    # Add all images with serial numbers
    for idx, im in enumerate(all_images):
        if im.path.exists():
            content.append({
                "type": "text",
                "text": f"[{idx}]"
            })
            with open(im.path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
            

    content.append({"type":"text", "text": "**GIVEN IMAGE**\n"})

    # Add the given image
    if target_image.path.exists():
        with open(target_image.path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": SPEAKER_RETRIEVAL},
        ]
    }]

    # Insert image-text pairs after the first text element
    messages[0]["content"] = [messages[0]["content"][0]] + content
    def __func_clean_up(gpt_response)->bool|dict:
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            num_list=gpt_response.get("num_list",[])
            if type(num_list) is list:
                if len(num_list) > 1:
                    return False
                for num in num_list:
                    if type(num) is not int or num < 0 or num >= len(all_images):
                        return False
                return gpt_response
            else:
                return False
                    
        except:
            return False

    cleaned_result = None
    result = ""
    attempts=3
    for i in range(attempts):
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
        cleaned_result = __func_clean_up(result)
        logger.info(
            "speaker_retrieval attempt: %s attempt=%s/%s valid=%s output=%s",
            log_context,
            i + 1,
            attempts,
            bool(cleaned_result),
            result,
        )
        if cleaned_result:
            break
    if not cleaned_result:
        logger.error("speaker_retrieval failed validation: %s output=%s", log_context, result)
        raise ValueError(f"speaker_retrieval returned invalid output after 3 attempts:\n{result}")
    logger.info("speaker_retrieval output: %s output=%s", log_context, result)
    return json.dumps(cleaned_result, ensure_ascii=False)

async def speaker_retrieval_async(
                    target_image:image,
                    all_images:List[image],
                    model:str="gpt-4o",
                    logger=None,
                    player_id=None,
                    phase=None,
                    round_num=None,
                    comm_round=None)->str:

    logger=_get_logger(logger)
    log_context=_log_context(player_id, phase, round_num, comm_round)
    logger.info(
        "speaker_retrieval_async started: %s target=%s all_images_count=%s",
        log_context,
        target_image,
        len(all_images),
    )
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }

    content = [{"type":"text", "text": "**ALL IMAGES**\n"}]

    for idx, im in enumerate(all_images):
        if im.path.exists():
            content.append({
                "type": "text",
                "text": f"[{idx}]"
            })
            with open(im.path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })

    content.append({"type":"text", "text": "**GIVEN IMAGE**\n"})

    if target_image.path.exists():
        with open(target_image.path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": SPEAKER_RETRIEVAL},
        ]
    }]

    messages[0]["content"] = [messages[0]["content"][0]] + content

    def __func_clean_up(gpt_response)->bool|dict:
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            num_list=gpt_response.get("num_list",[])
            if type(num_list) is list:
                if len(num_list) > 1:
                    return False
                for num in num_list:
                    if type(num) is not int or num < 0 or num >= len(all_images):
                        return False
                return gpt_response
            else:
                return False
                    
        except:
            return False

    cleaned_result = None
    result = ""
    retryable_status_codes = {408, 409, 429, 500, 502, 503, 504}
    max_attempts = 6
    base_delay = 2
    for i in range(max_attempts):
        try:
            await asyncio.sleep(0.5)
            response = await async_client.chat.completions.create(
                model=gpt_param["model"],
                messages=messages,
                max_tokens=gpt_param["max_tokens"],
                top_p=gpt_param["top_p"],
                frequency_penalty=gpt_param["frequency_penalty"],
                presence_penalty=gpt_param["presence_penalty"],
                temperature=gpt_param["temperature"],
            )
            result = response.choices[0].message.content
            cleaned_result = __func_clean_up(result)
            logger.info(
                "speaker_retrieval_async attempt: %s attempt=%s/%s valid=%s output=%s",
                log_context,
                i + 1,
                max_attempts,
                bool(cleaned_result),
                result,
            )
            if cleaned_result:
                break

            delay = base_delay * (2 ** i) + random.uniform(0, 1)
            logger.warning(
                "speaker_retrieval_async failed validation: %s attempt=%s/%s target=%s retry_in=%.2fs output=%s",
                log_context,
                i + 1,
                max_attempts,
                target_image,
                delay,
                result,
            )
            await asyncio.sleep(delay)
        except openai.APIStatusError as e:
            if e.status_code not in retryable_status_codes or i == max_attempts - 1:
                raise
            delay = base_delay * (2 ** i) + random.uniform(0, 1)
            logger.warning(
                "speaker_retrieval_async API status error: %s attempt=%s/%s target=%s status=%s retry_in=%.2fs",
                log_context,
                i + 1,
                max_attempts,
                target_image,
                e.status_code,
                delay,
            )
            await asyncio.sleep(delay)
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            if i == max_attempts - 1:
                raise
            delay = base_delay * (2 ** i) + random.uniform(0, 1)
            logger.warning(
                "speaker_retrieval_async API connection error: %s attempt=%s/%s target=%s retry_in=%.2fs error=%s",
                log_context,
                i + 1,
                max_attempts,
                target_image,
                delay,
                e,
            )
            await asyncio.sleep(delay)
    if not cleaned_result:
        logger.error("speaker_retrieval_async failed validation: %s output=%s", log_context, result)
        raise ValueError(f"speaker_retrieval_async returned invalid output after {max_attempts} attempts:\n{result}")
    logger.info("speaker_retrieval_async output: %s output=%s", log_context, result)
    return json.dumps(cleaned_result, ensure_ascii=False)

def listener_retrieval(
    word:str,
    learned_vocabulary:List[str],
    letters:List[str],
    model:str,
    logger=None,
    player_id=None,
    phase=None,
    round_num=None,
    comm_round=None,
)->dict:
    

    logger=_get_logger(logger)
    log_context=_log_context(player_id, phase, round_num, comm_round)
    logger.info(
        "listener_retrieval started: %s word=%s vocab_size=%s",
        log_context,
        word,
        len(learned_vocabulary),
    )
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }


    # Format the prompt with inputs
    prompt_inputs = [
        str(len(letters)),                          # !<INPUT 0>! - number of letters
        list_to_table(letters,"Letters"), 
        list_to_table(learned_vocabulary, "Learned Vocabulary"),
        word,                          
    ]
    formatted_prompt = LISTENER_RETRIEVAL
    for idx, inp in enumerate(prompt_inputs):
        formatted_prompt = formatted_prompt.replace(f"!<INPUT {idx}>!", inp)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": formatted_prompt},
        ]
    }]

    def __func_clean_up(gpt_response,
                        vocab=learned_vocabulary)->bool|dict:
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            num_list=gpt_response.get("num_list",[])
            if type(num_list) is not list:
                return False
            for num in num_list:
                if type(num) is not int or num < 1 or num > len(vocab):
                    return False
            word_list=[]
            for num in num_list:
                word_list.append(vocab[num-1])
            gpt_response["word_list"]=word_list
            return gpt_response            
        except:
            return False

    cleaned_result = None
    result = ""
    attempts=5
    for attempt in range(attempts):
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
        cleaned_result = __func_clean_up(result)
        logger.info(
            "listener_retrieval attempt: %s attempt=%s/%s valid=%s output=%s",
            log_context,
            attempt + 1,
            attempts,
            bool(cleaned_result),
            result,
        )
        if cleaned_result:
            break
    if not cleaned_result:
        logger.error("listener_retrieval failed validation: %s output=%s", log_context, result)
        raise ValueError(f"listener_retrieval returned invalid output after {attempts} attempts:\n{result}")
    logger.info("listener_retrieval output: %s output=%s", log_context, result)
    return cleaned_result
