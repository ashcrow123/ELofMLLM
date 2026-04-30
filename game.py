from communicator.communicator import communicator
import random
import os
import json
from tqdm import tqdm
from dataclasses import dataclass,asdict
from model import all_images, image


@dataclass
class gameconfig:
    name:str
    player_num:int
    letter_list:list
    comm_num:int
    model_list:list
    max_length:int
    option_num:int


def select_letters(num=10,seed=None):
    random.seed(seed)
    all_letters=[]
    for i in "bcdfghjklmnpqrstvwxyz":
        for j in "aeiou":
            all_letters.append(i+j)
    letters=random.sample(all_letters,k=num)
    random.seed(None)
    return letters


def select_letters_vcv(num=10,seed=None):
    random.seed(seed)
    all_letters=[]
    for i in "aeiou":
        for j in "bcdfghjklmnpqrstvwxyz":
            for k in "aeiou":
                all_letters.append(i+j+k)
    letters=random.sample(all_letters,k=num)
    random.seed(None)
    return letters


def select_letters_num(num=10,seed=None):
    letters=[]
    for i in range(1,num+1):
        letters.append(f"({i})")
    return letters


def select_letters_nl(num=10,seed=None):
    random.seed(seed)
    all_letters=[]
    for i in "0123456789":
        for j in "abcdefghijklmnopqrstuvwxyz":
            all_letters.append(f"{i}{j}")
    letters=random.sample(all_letters,k=num)
    random.seed(None)
    return letters


def load_object_image_pairs(image_pool=None)->dict:
    if image_pool is None:
        image_pool=all_images
    pairs={}
    for item in image_pool:
        if isinstance(item, image):
            im=item
        else:
            im=image(num=int(item))
        pairs[im.num]=im
    return pairs


def get_splited_data(train_ratio=0.8, seed=None, image_pool=None, count=None):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    pairs=load_object_image_pairs(image_pool)
    keys=list(pairs.keys())
    random.seed(seed)
    random.shuffle(keys)
    if count is not None:
        count=min(count, len(keys))
        keys=keys[:count]

    train_count=max(1, min(len(keys)-1, round(len(keys)*train_ratio)))
    train_keys=set(keys[:train_count])
    train_pairs={key:pairs[key] for key in keys if key in train_keys}
    test_pairs={key:pairs[key] for key in keys if key not in train_keys}
    random.seed(None)
    return train_pairs,test_pairs


class Referential_Game:
    def __init__(
        self,
        name:str,
        player_num:int,
        letter_list:list,
        comm_num,
        save_interval,
        obj_loader=None,
        max_length=6,
        model_list=None,
        option_num=5,
        logger=None,
    ):
        self.logger=logger
        self.name=name
        self.comm_num=comm_num
        if player_num<2 or player_num%2!=0:
            raise ValueError("The number of players must be an even number and greater than 1.")
        if option_num < 2 or option_num > 5:
            raise ValueError("option_num must be between 2 and 5 because listener choices are labeled A-E.")
        if model_list is None:
            raise ValueError("model_list must be provided.")
        if len(model_list) != player_num:
            raise ValueError("model_list length must match player_num.")

        self.max_length=max_length
        self.player_num=player_num
        self.letter_list=letter_list
        self.model_list=model_list
        self.players=dict()
        for i in range(player_num):
            self.players[str(i)]=communicator(
                self.letter_list,
                id=str(i),
                max_length=self.max_length,
                model=self.model_list[i],
                logger=self.logger,
            )
        self.round=0
        self.obj_loader=obj_loader if obj_loader is not None else {im.num: im for im in all_images}
        self.save_interval=save_interval
        self.option_num=option_num
        config=gameconfig(
            name=name,
            player_num=player_num,
            letter_list=letter_list,
            comm_num=comm_num,
            model_list=model_list,
            max_length=max_length,
            option_num=option_num,
        )
        os.makedirs(f"./sim_storage/{self.name}",exist_ok=True)
        if not os.path.exists(f"./sim_storage/{self.name}/game_config.json"):
            with open(f"./sim_storage/{self.name}/game_config.json","w") as f:
                json.dump(asdict(config),f,indent=4)
        else:
            with open(f"./sim_storage/{self.name}/game_config.json","r") as f:
                old_config=json.load(f)
            if old_config!=asdict(config):
                raise ValueError("The game configuration is inconsistent with the historical configuration.")
        if self.logger:
            self.logger.info(
                "game initialized: name=%s player_num=%s comm_num=%s option_num=%s objects=%s",
                self.name,
                self.player_num,
                self.comm_num,
                self.option_num,
                len(self._obj_keys()),
            )

    def _obj_keys(self):
        if isinstance(self.obj_loader, dict):
            return list(self.obj_loader.keys())
        return list(range(len(self.obj_loader)))

    def _to_image(self, obj_key)->image:
        if isinstance(obj_key, image):
            return obj_key
        if isinstance(self.obj_loader, dict):
            obj_value=self.obj_loader[obj_key]
            if isinstance(obj_value, image):
                return obj_value
            if isinstance(obj_value, int):
                return image(num=obj_value)
        elif isinstance(self.obj_loader[obj_key], image):
            return self.obj_loader[obj_key]
        elif isinstance(self.obj_loader[obj_key], int):
            return image(num=self.obj_loader[obj_key])
        return image(num=int(obj_key))

    def _round_target_key(self):
        obj_keys=self._obj_keys()
        return obj_keys[(self.round-1)%len(obj_keys)]

    def _sample_distractor_keys(self, corr_key):
        keys_list=self._obj_keys()
        keys_list.remove(corr_key)
        return random.sample(keys_list,k=self.option_num-1)

    def communicate(
        self,
        speaker_id:str,
        listener_id:str,
        obj_dict:dict,
        phase="run",
    ):
        speaker=self.players[speaker_id]
        listener=self.players[listener_id]
        round_num=0
        success=False
        word_list=[]
        choices_list=[]
        corr_key=self._round_target_key()
        corr_image=self._to_image(corr_key)
        choice_images=[self._to_image(key) for key in obj_dict.keys()] + [corr_image]
        random.shuffle(choice_images)
        choice_labels=["A", "B", "C", "D", "E"]
        choice_map={choice_labels[i]: obj.num for i, obj in enumerate(choice_images)}
        if self.logger:
            self.logger.info(
                "choice map: phase=%s round=%s speaker_id=%s listener_id=%s target=%s choice_map=%s",
                phase,
                self.round,
                speaker_id,
                listener_id,
                corr_image.num,
                choice_map,
            )

        speaker_known_vocab_num=speaker.word_database.search_word(corr_image)
        random.shuffle(speaker_known_vocab_num)
        speaker_known_vocab=[speaker.word_database.word_dict[num].word for num in speaker_known_vocab_num]
        speaker_near_synonym_nums=speaker.word_database.search_near_synonyms(corr_image)
        speaker_near_synonyms=speaker.word_database.get_word_image_dict(speaker_near_synonym_nums)

        while True:
            round_num+=1
            if round_num>self.comm_num:
                break

            used_speaker_num=None
            if speaker_known_vocab_num:
                used_speaker_num=speaker_known_vocab_num.pop(0)
                word=speaker.word_database.word_dict[used_speaker_num].word
            else:
                word, speaker_source=speaker.generate_new_word(
                    target_object=corr_image,
                    learned_vocabulary=speaker_near_synonyms,
                    phase=phase,
                    round_num=self.round,
                    comm_round=round_num,
                )
            if used_speaker_num is not None:
                speaker_source="local_known_word"

            word_exists,selected_image_num,used_listener_num,listener_source=listener.listener_select(
                word=word,
                target_object=corr_image,
                choices=choice_images,
                phase=phase,
                round_num=self.round,
                comm_round=round_num,
            )
            choices_list.append(selected_image_num)
            word_list.append(word)
            if self.logger:
                self.logger.info(
                    "communication step: phase=%s round=%s comm_round=%s speaker_id=%s listener_id=%s word=%s target=%s selected=%s word_exists=%s speaker_source=%s listener_source=%s choice_map=%s",
                    phase,
                    self.round,
                    round_num,
                    speaker_id,
                    listener_id,
                    word,
                    corr_image.num,
                    selected_image_num,
                    word_exists,
                    speaker_source,
                    listener_source,
                    choice_map,
                )

            if selected_image_num==corr_image.num:
                success=True
                if used_speaker_num is None:
                    speaker.word_database.add_word(word=word, obj=corr_image)
                if not word_exists:
                    listener_word_list=listener.word_database.search_word(corr_image)
                    if listener_word_list:
                        listener.word_database.change_word(listener_word_list[0],word)
                    else:
                        listener.word_database.add_word(word=word, obj=corr_image)
                break

        return {
            "speaker_id":speaker_id,
            "listener_id":listener_id,
            "success":success,
            "word_list":word_list,
            "obj_list":[obj.num for obj in choice_images],
            "choice_map":choice_map,
            "corr_obj":corr_image.num,
            "choices_list":choices_list,
            "speaker_near_synonyms":list(speaker_near_synonyms.keys()),
            "speaker_known_vocab":speaker_known_vocab
        }

    def test_communicate(
        self,
        speaker_id:str,
        listener_id:str,
        obj_dict:dict,
        phase="test_run",
    ):
        speaker=self.players[speaker_id]
        listener=self.players[listener_id]
        round_num=0
        success=False
        word_list=[]
        choices_list=[]
        corr_key=self._round_target_key()
        corr_image=self._to_image(corr_key)
        choice_images=[self._to_image(key) for key in obj_dict.keys()] + [corr_image]
        random.shuffle(choice_images)
        choice_labels=["A", "B", "C", "D", "E"]
        choice_map={choice_labels[i]: obj.num for i, obj in enumerate(choice_images)}
        if self.logger:
            self.logger.info(
                "choice map: phase=%s round=%s speaker_id=%s listener_id=%s target=%s choice_map=%s",
                phase,
                self.round,
                speaker_id,
                listener_id,
                corr_image.num,
                choice_map,
            )

        speaker_near_synonym_nums=speaker.word_database.search_near_synonyms(corr_image)
        speaker_near_synonyms=speaker.word_database.get_word_image_dict(speaker_near_synonym_nums)
        speaker_known_vocab_num=speaker.word_database.search_word(corr_image)
        random.shuffle(speaker_known_vocab_num)
        speaker_known_vocab=[speaker.word_database.word_dict[num].word for num in speaker_known_vocab_num]

        while True:
            round_num+=1
            if round_num>self.comm_num:
                break

            if speaker_known_vocab_num:
                word_num=speaker_known_vocab_num.pop(0)
                word=speaker.word_database.word_dict[word_num].word
                speaker_source="local_known_word"
            else:
                word, speaker_source=speaker.generate_new_word(
                    target_object=corr_image,
                    learned_vocabulary=speaker_near_synonyms,
                    phase=phase,
                    round_num=self.round,
                    comm_round=round_num,
                )

            _,selected_image_num,_,listener_source=listener.listener_select(
                word=word,
                target_object=corr_image,
                choices=choice_images,
                phase=phase,
                round_num=self.round,
                comm_round=round_num,
            )
            choices_list.append(selected_image_num)
            word_list.append(word)
            if self.logger:
                self.logger.info(
                    "communication step: phase=%s round=%s comm_round=%s speaker_id=%s listener_id=%s word=%s target=%s selected=%s speaker_source=%s listener_source=%s choice_map=%s",
                    phase,
                    self.round,
                    round_num,
                    speaker_id,
                    listener_id,
                    word,
                    corr_image.num,
                    selected_image_num,
                    speaker_source,
                    listener_source,
                    choice_map,
                )
            if selected_image_num==corr_image.num:
                success=True
                break

        return {
            "speaker_id":speaker_id,
            "listener_id":listener_id,
            "success":success,
            "word_list":word_list,
            "obj_list":[obj.num for obj in choice_images],
            "choice_map":choice_map,
            "corr_obj":corr_image.num,
            "choices_list":choices_list,
            "speaker_near_synonyms":list(speaker_near_synonyms.keys()),
            "speaker_known_vocab":speaker_known_vocab
        }

    def run(self,rounds):
        if self.logger:
            self.logger.info("train run started: rounds=%s start_round=%s", rounds, self.round)
        save_flag=0
        for _ in tqdm(range(rounds)):
            save_flag+=1
            results=[]
            self.round+=1
            player_ids=[str(i) for i in range(self.player_num)]
            random.shuffle(player_ids)
            pairs=[]
            for i in range(0,self.player_num,2):
                pairs.append((player_ids[i],player_ids[i+1]))
            print(pairs)
            for pair in pairs:
                if self.logger:
                    self.logger.info("round=%s pair=%s mode=train", self.round, pair)
                corr_key=self._round_target_key()
                random_keys=self._sample_distractor_keys(corr_key)
                obj_dict={obj: self.obj_loader[obj] for obj in random_keys}
                result=self.communicate(
                    speaker_id=pair[0],
                    listener_id=pair[1],
                    obj_dict=obj_dict,
                    phase="run",
                )
                results.append(result)
                if self.logger:
                    self.logger.info("round=%s result=%s", self.round, result)
            if save_flag%self.save_interval==0:
                self.save(results,True)
            else:
                self.save(results,False)

    def test_run(self,rounds):
        if self.logger:
            self.logger.info("test run started: rounds=%s start_round=%s", rounds, self.round)
        for _ in tqdm(range(rounds)):
            results=[]
            self.round+=1
            player_ids=[str(i) for i in range(self.player_num)]
            random.shuffle(player_ids)
            pairs=[]
            for i in range(0,self.player_num,2):
                pairs.append((player_ids[i],player_ids[i+1]))
            print(pairs)
            for pair in pairs:
                if self.logger:
                    self.logger.info("round=%s pair=%s mode=test", self.round, pair)
                corr_key=self._round_target_key()
                random_keys=self._sample_distractor_keys(corr_key)
                obj_dict={obj: self.obj_loader[obj] for obj in random_keys}
                result=self.test_communicate(
                    speaker_id=pair[0],
                    listener_id=pair[1],
                    obj_dict=obj_dict,
                    phase="test_run",
                )
                results.append(result)
                if self.logger:
                    self.logger.info("round=%s result=%s", self.round, result)
            self.save(results,False)

    def save(self,results,with_worddatabase):
        round_path=f"./sim_storage/{self.name}/round_{self.round}"
        try:
            os.makedirs(round_path,exist_ok=False)
        except:
            raise FileExistsError("The file for this round of the game already exists.")
        os.makedirs(os.path.join(round_path,"Communicator_Worddatabase"))
        with open(os.path.join(round_path,"results.json"),"w") as f:
            json.dump(results,f,indent=4)
        if with_worddatabase:
            for num in list(self.players.keys()):
                os.makedirs(os.path.join(round_path,"Communicator_Worddatabase",f"Player_{str(num)}"))
                self.players[num].save(
                    os.path.join(round_path,"Communicator_Worddatabase",f"Player_{str(num)}")
                )
        print(f"round_{self.round} has saved.")
        if self.logger:
            self.logger.info("round_%s saved: path=%s with_worddatabase=%s", self.round, round_path, with_worddatabase)

    def load(self,round):
        if os.path.exists(f"./sim_storage/{self.name}/round_{round}"):
            pass
        else:
            raise FileExistsError('The folder for this round of the game does not exist.')
        self.round=round
        player_num=len(os.listdir(f"./sim_storage/{self.name}/round_{round}/Communicator_Worddatabase"))
        if player_num!=self.player_num:
            raise ValueError("The number of players set in the game is different from the historical number")
        for key in list(self.players.keys()):
            player_path=f"./sim_storage/{self.name}/round_{round}/Communicator_Worddatabase/Player_{str(key)}"
            self.players[key].load(player_path)
        print(f"round_{self.round} has loaded.")
        if self.logger:
            self.logger.info("round_%s loaded", self.round)
