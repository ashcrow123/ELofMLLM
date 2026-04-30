from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel
@dataclass
class image:
    num:int 
    @property
    def path(self)->Path:
        return Path(f"./abstract_image/images/orchid_{self.num:03d}.png")
all_images = [image(num=i) for i in range(100)]   
 
@dataclass
class object_network:
    network:Dict[str, List[image]]
    def get_near_synonyms(self,image_num:int)->List[image]:
        image_key = str(image_num)
        if image_key not in self.network:
            return []
        return self.network[image_key]
    
@dataclass
class Word:
    obj: int
    word: str
    speak_fail_count: int = 0
    listen_fail_count: int = 0

    @property
    def todict(self)->dict:
        return {
            "obj": self.obj,
            "word": self.word,
            "speak_fail_count": self.speak_fail_count,
            "listen_fail_count": self.listen_fail_count,
        }

    @property
    def todict_wo_object(self)->dict:
        return {
            "word": self.word,
            "image": image(num=self.obj),
            "image_num": self.obj,
        }

    @property
    def toFeatures(self)->dict:
        return {"obj": self.obj}

    def change_word(self, word: str):
        self.word=word
