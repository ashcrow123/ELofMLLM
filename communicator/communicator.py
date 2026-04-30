import json
import random
from typing import Dict, List

from communicator.WordDatabase import WordDatabase
from llm_provider.run_gpt_prompt import listener_retrieval, listener_select, speaker_generate
from model import image


class communicator:
    def __init__(self, letter_list, id, max_length, model, logger=None):
        self.letter_list=letter_list
        self.word_database=WordDatabase(model=model)
        self.player_id=id
        self.max_length=max_length
        self.model=model
        self.logger=logger

    def save(self,path):
        self.word_database.save(path)

    def load(self,path):
        self.word_database.load(path)
        print(f"player_{self.player_id} has loaded word database.")

    def generate_new_word(
        self,
        target_object: image,
        learned_vocabulary: Dict[str, image],
        phase=None,
        round_num=None,
        comm_round=None,
    )->str:
        if learned_vocabulary:
            result=speaker_generate(
                target_object=target_object,
                learned_vocabulary=learned_vocabulary,
                letters=self.letter_list,
                max_word_len=self.max_length,
                model=self.model,
                logger=self.logger,
                player_id=self.player_id,
                phase=phase,
                round_num=round_num,
                comm_round=comm_round,
            )
            return json.loads(result)["word"], "llm_speaker_generate"

        if self.max_length==2:
            length=random.randint(1,self.max_length)
        else:
            length=random.randint(2,self.max_length)
        return "-".join(random.choices(self.letter_list,k=length)), "random_new_word"

    def listener_select(
        self,
        word: str,
        target_object: image,
        choices: List[image],
        phase=None,
        round_num=None,
        comm_round=None,
    ):
        word_exists=False
        choice_nums={obj.num for obj in choices}
        if word in self.word_database.word_to_key_dict:
            candidate_pairs=[]
            for num in self.word_database.word_to_key_dict[word]:
                item=self.word_database.word_dict[num]
                if item.obj in choice_nums:
                    candidate_pairs.append((item.obj, num))
            if candidate_pairs:
                word_exists=True
                selected_num, used_num=random.choice(candidate_pairs)
                return word_exists, selected_num, used_num, "local_exact_word"

        learned_vocabulary=self.word_database.get_word_image_dict()
        source="llm_listener_select_full_vocab"
        if learned_vocabulary:
            resembling_result=listener_retrieval(
                word=word,
                learned_vocabulary=list(learned_vocabulary.keys()),
                letters=self.letter_list,
                model=self.model,
                logger=self.logger,
                player_id=self.player_id,
                phase=phase,
                round_num=round_num,
                comm_round=comm_round,
            )
            resembling_words=resembling_result["word_list"]
            if resembling_words:
                source="llm_listener_select_resembling_vocab"
                learned_vocabulary={
                    resem_word: learned_vocabulary[resem_word]
                    for resem_word in resembling_words
                    if resem_word in learned_vocabulary
                }

        result=listener_select(
            word=word,
            target_object=target_object,
            learned_vocabulary=learned_vocabulary,
            letters=self.letter_list,
            max_word_len=self.max_length,
            model=self.model,
            choices=choices,
            logger=self.logger,
            player_id=self.player_id,
            phase=phase,
            round_num=round_num,
            comm_round=comm_round,
        )
        selected_image_num=json.loads(result)["selected_image_num"]
        return word_exists, selected_image_num, None, source
