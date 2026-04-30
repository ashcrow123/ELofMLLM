import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from model import image,Word


def split_cv_blocks(word: str)->List[str]:
    return word.split("-")


def Jaccard_similarity(blocks_1: List[str], blocks_2: List[str], n: int)->float:
    def n_grams(blocks: List[str], n: int):
        return set(tuple(blocks[i:i+n]) for i in range(len(blocks) - n + 1))

    set1=n_grams(blocks_1, n)
    set2=n_grams(blocks_2, n)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0




class WordDatabase:
    def __init__(self, model: str):
        self.word_dict: Dict[str, Word]=dict()
        self.word_to_key_dict: Dict[str, List[str]]=dict()
        self.obj_dict: Dict[str, List[str]]=dict()
        self.synonyms_search_dict=self._load_object_network(model)

    def _load_object_network(self, model: str)->dict:
        model_name=model.replace("/", "-")
        candidate_paths=[
            Path("object_network") / f"{model_name}_network.json",
            Path("data") / f"{model_name}_network.json",
        ]
        for path in candidate_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        warnings.warn(
            f"No object network found for model {model}. Near-synonym search will be empty.",
            RuntimeWarning,
        )
        return {}

    def _obj_key(self, obj)->str:
        if isinstance(obj, image):
            return str(obj.num)
        return str(obj)

    def _obj_num(self, obj)->int:
        if isinstance(obj, image):
            return obj.num
        return int(obj)

    def add_word(self, word: str, obj, text_embedding=None, **_ignored):
        obj_num=self._obj_num(obj)
        obj_key=str(obj_num)
        if word in self.get_word_list():
            for num in self.word_to_key_dict[word]:
                if self.word_dict[num].obj == obj_num:
                    warnings.warn(f"The word {word} has already in the word database.", RuntimeWarning)
                    return

        new_word=Word(obj=obj_num, word=word)
        new_num=str(np.max([int(num) for num in self.word_dict.keys()])+1) if self.word_dict else "0"
        self.word_dict[new_num]=new_word
        self.word_to_key_dict.setdefault(word, []).append(new_num)
        self.obj_dict.setdefault(obj_key, []).append(new_num)

    def change_word(self, num, word: str):
        num=str(num)
        old_word=self.word_dict[num].word
        if old_word==word:
            return
        self.word_dict[num].change_word(word)
        self.word_to_key_dict[old_word].remove(num)
        if not self.word_to_key_dict[old_word]:
            del self.word_to_key_dict[old_word]
        self.word_to_key_dict.setdefault(word, []).append(num)

    def search_word(self, obj=None, **_ignored)->List[str]:
        if obj is None:
            return []
        return list(self.obj_dict.get(self._obj_key(obj), []))

    def search_near_synonyms(self, obj)->List[str]:
        obj_key=self._obj_key(obj)
        synonyms_list=self.synonyms_search_dict.get(obj_key, [])
        num_list=[]
        for syn_obj in synonyms_list:
            syn_key=str(syn_obj)
            if syn_key in self.obj_dict:
                num_list+=self.obj_dict[syn_key]
        own_nums=set(self.obj_dict.get(obj_key, []))
        return [num for num in num_list if num not in own_nums]

    def search_resembling_word(self, target_word: str)->List[str]:
        target_blocks = split_cv_blocks(target_word)
        distances = []
        for word in self.word_to_key_dict.keys():
            if word!=target_word:
                try:
                    blocks = split_cv_blocks(word)
                    similarity=(
                        Jaccard_similarity(target_blocks, blocks, n=2)
                        + Jaccard_similarity(target_blocks, blocks, n=3)
                        + Jaccard_similarity(target_blocks, blocks, n=4)
                        + Jaccard_similarity(target_blocks, blocks, n=5)
                    )
                    if similarity>0:
                        distances.append(word)
                except Exception:
                    continue
        random.shuffle(distances)
        return distances

    def get_word_image_dict(self, nums: List[str] | None = None)->Dict[str, image]:
        if nums is None:
            nums=list(self.word_dict.keys())
        learned_vocabulary={}
        for num in nums:
            item=self.word_dict[str(num)]
            learned_vocabulary[item.word]=image(num=item.obj)
        return learned_vocabulary

    def weight_output(self, num_list, identity, beta=1.3):
        if not (identity in ["speaker","listener"]):
            raise ValueError("identity must be 'speaker' or 'listener'. ")
        if not num_list:
            return None
        num_list=[str(num) for num in num_list]
        if not set(num_list).issubset(set(self.word_dict.keys())):
            raise ValueError("There are numbers in the list that are not in the word database.")
        total=0
        weights=[]
        for num in num_list:
            if identity=="speaker":
                word_fail_count=self.word_dict[num].speak_fail_count
            else:
                word_fail_count=self.word_dict[num].listen_fail_count
            exp_num=np.exp(-word_fail_count*beta)
            total+=exp_num
            weights.append(exp_num)
        weights=[weight/total for weight in weights]
        return weights

    def load(self, path):
        with open(os.path.join(path,"word_dict.json"),"r",encoding="utf-8") as f:
            load_dict=json.load(f)
        self.word_dict={key: Word(**value) for key, value in load_dict.items()}
        with open(os.path.join(path,"word_to_key_dict.json"),"r",encoding="utf-8") as f:
            self.word_to_key_dict=json.load(f)
        with open(os.path.join(path,"obj_dict.json"),"r",encoding="utf-8") as f:
            self.obj_dict=json.load(f)

    def save(self, path):
        os.makedirs(path,exist_ok=True)
        save_dict={key: word.todict for key, word in self.word_dict.items()}
        with open(os.path.join(path,"word_dict.json"),"w",encoding="utf-8") as f:
            json.dump(save_dict,f,ensure_ascii=False,indent=4)
        with open(os.path.join(path,"word_to_key_dict.json"),"w",encoding="utf-8") as f:
            json.dump(self.word_to_key_dict,f,ensure_ascii=False,indent=4)
        with open(os.path.join(path,"obj_dict.json"),"w",encoding="utf-8") as f:
            json.dump(self.obj_dict,f,ensure_ascii=False,indent=4)

    def get_word_list(self)->List[str]:
        return list(self.word_to_key_dict.keys())

    def delete(self, num):
        num=str(num)
        if num not in self.word_dict:
            warnings.warn(f"The num {num} is not in the word database.",RuntimeWarning)
            return
        del self.word_dict[num]
        word_remove_list=[]
        for word,num_list in self.word_to_key_dict.items():
            if num in num_list:
                num_list.remove(num)
                if not num_list:
                    word_remove_list.append(word)
        for word in word_remove_list:
            del self.word_to_key_dict[word]

        obj_remove_list=[]
        for obj,num_list in self.obj_dict.items():
            if num in num_list:
                num_list.remove(num)
                if not num_list:
                    obj_remove_list.append(obj)
        for obj in obj_remove_list:
            del self.obj_dict[obj]
