import heapq
import random
import numpy as np
import tensorflow as tf

from typing import List, Any, Dict, Callable, Set, Tuple

import importlib.util  

from os import listdir
from os.path import join, abspath, dirname

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

from nltk import download, word_tokenize
from nltk.stem import WordNetLemmatizer
    
from json import load

from numpy import ndarray

from warnings import filterwarnings

from kato_chat_bot.model.model_version import Model_Version
from kato_chat_bot.model.model_settings.settings import PACKAGES

CURRENT_DIRECTORY = dirname(abspath(__file__))
MODEL_DIRECTORY: str = join(CURRENT_DIRECTORY, "saved_models")

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

class Model_Handler:
    def __init__(self):
        self.available_versions: List[Model_Version] = self._get_available_model_list(
        )
        self.model_generator_params: Dict[str, Callable] = {
            "major": self._get_next_major_version,
            "minor": self._get_next_minor_version,
            "patch": self._get_next_patch_version
        }

    def get_latest_model_version(self) -> Model_Version:
        latest_version: Model_Version = heapq.heappop(self.available_versions)
        heapq.heappush(self.available_versions, latest_version)

        return latest_version

    def get_model(model_version: Model_Version) -> Any:
        model_version_string: str = str(model_version)

        return load_model(f"saved_model/model_v_{model_version_string}")

    def model_version_exists(self, version: Model_Version) -> bool:
        return version in self.available_versions


    def create_and_save_model(self, version_param: str, intents: dict, function_mapper_file_bytes: any) -> None:
        version: Model_Version = Model_Version(version_param)
        
        if self.model_version_exists(version):
            return 

        if not self._download_packages():
            return

        intent_data = self._process_intents(intents)

        classes, words, document = intent_data["classes"], intent_data["words"], intent_data["document"]
        
        model = self._create_model(classes, words, document)
        
        model_path: str = f'{MODEL_DIRECTORY}/{str(version)}'
        
        model.save(f"{model_path}/model")
        
        self._save_model_misc(intent_data["classes"], f'{model_path}/classes.txt')
        self._save_model_misc(intent_data["words"], f'{model_path}/words.txt')
        self._save_model_misc(intents, f'{model_path}/intents.json')
                
        with open(f'{model_path}/function_mapper.py', "wb") as binary_file:     
            binary_file.write(function_mapper_file_bytes)


    def load_model_and_misc(self, version: Model_Version) -> tuple:
        intents: dict = self._load_intents(version)
        classes: List[str] = self._load_model_misc(version, "classes.txt")
        words: List[str] = self._load_model_misc(version, "words.txt")
                
        return load_model(f"{MODEL_DIRECTORY}/{str(version)}/model"), intents, classes, words
    
    
    def get_result_from_input(self, user_input: str, model: any, intents: dict, classes: List[str], words: List[str], version: Model_Version) -> str:
        response: List[str] = self._get_model_response(model, user_input, words, classes)
        intent = self._get_result(response, intents)
        
        if intent == "NO INTENT FOUND":
            return "Sorry, I do not know how to respond to that."
        
        # get the functin mapper for this version
        function_mapper_file_spec = importlib.util.spec_from_file_location("function_mapper", f"{MODEL_DIRECTORY}/{str(version)}/function_mapper.py")
        function_mapper_lib = importlib.util.module_from_spec(function_mapper_file_spec)
        
        function_mapper_file_spec.loader.exec_module(function_mapper_lib)

        print(function_mapper_lib)
        return function_mapper_lib.map_function(user_input, intent)


    # PRIVATE FUNCTIONS START HERE
    def _download_packages(self) -> bool:
        package_count: int = len(PACKAGES)

        for i, package in enumerate(PACKAGES):
            try:
                download(package, quiet=True)
                print(f"{i+1}/{package_count}: Downloaded {package}")
            except:
                print(f"Could not download {package}")
                return False

        print("Downloaded all packages")
        return True

    def _process_intents(self, intents: dict) -> dict:
        lemmatizer = WordNetLemmatizer()

        classes: Set[str] = set()
        words: Set[str] = set()
        doc_x, doc_y = [], []
        
        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                tokens: List[str] = word_tokenize(pattern) #extract words from each pattern
                tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

                words.update(tokens)

                doc_x.append(pattern)
                doc_y.append(intent["tag"])

            classes.add(intent["tag"])

        return {
            "classes"   : sorted(classes),
            "words"     : sorted(words),
            "document"  : {
                "x": doc_x,
                "y": doc_y
            }
        }

    def _create_model(self, classes: str, words: str, documents: Dict[str, List[str]]) -> any:
        document_x, document_y = documents["x"], documents["y"]
        lemmatizer = WordNetLemmatizer()

        training_data = []
        out_empty: List[int] = [0] * len(classes)

        for index, document in enumerate(document_x):
            bag_of_words = []
            text = lemmatizer.lemmatize(document.lower())

            for word in words:
                bag_of_words.append(1) if word in text else bag_of_words.append(0)

            output_row = list(out_empty)
            output_row[classes.index(document_y[index])] = 1
            training_data.append([bag_of_words, output_row])


        random.shuffle(training_data)
        training_data = np.array(training_data, dtype = object)

        x = np.array(list(training_data[:, 0])) # first training phase
        y = np.array(list(training_data[:, 1])) # second training phase

        i_shape = (len(x[0]), )
        o_shape = len(y[0])

        model = Sequential()
        model.add(Dense(128, input_shape = i_shape, activation = "relu"))
        model.add(Dropout(0.5))

        model.add(Dense(64, activation = "relu"))
        model.add(Dropout(0.3))
        model.add(Dense(o_shape, activation = "softmax"))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=1e-6
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

        model.compile(
            loss        = 'categorical_crossentropy',
            optimizer   = optimizer,
            metrics     = ["accuracy"]
        )

        model.fit(x, y, epochs = 1000, verbose = 1)

        return model

    def _save_model_misc(self, data: any, file_name: str) -> None:
        data_file = open(file_name, "w")

        for element in data:
            data_file.write(element + "\n")

        data_file.close
        
    def _load_model_misc(self, version: Model_Version, file_name: str) -> List[str]:
        data: List[str] = []

        with open(f"{MODEL_DIRECTORY}/{str(version)}/{file_name}", "r") as file:
            for line in file:
                data.append(line.strip())

        return data
    
    
    def _load_intents(self, version: Model_Version) -> dict:
        data = {}

        with open(f"{MODEL_DIRECTORY}/{str(version)}/intents.json", "r") as file:
            data = load(file)
            
        return data

    def _get_available_model_list(self) -> List[Model_Version]:
        model_versions: List[Model_Version] = []

        for file_name in listdir(MODEL_DIRECTORY):
            if file_name[:5] == "model":
                model_versions.append(self._extract_model_version(file_name))

        heapq._heapify_max(model_versions)

        return model_versions

    def _extract_model_version(self, file_name: str) -> Model_Version:
        version_string: str = file_name[6:]

        return Model_Version(version=version_string)

    def _get_next_major_version(self) -> Model_Version:
        latest_version_dict: Dict[str, int] = self.get_latest_model_version(
        ).convertToDict()

        latest_version_dict["major"] += 1
        latest_version_dict["minor"] = 0
        latest_version_dict["patch"] = 0

        return Model_Version(latest_version_dict)

    def _get_next_minor_version(self) -> Model_Version:
        latest_version_dict: Dict[str, int] = self.get_latest_model_version(
        ).convertToDict()

        latest_version_dict["minor"] += 1
        latest_version_dict["patch"] = 0

        return Model_Version(latest_version_dict)

    def _get_next_patch_version(self) -> Model_Version:
        latest_version_dict: Dict[str, int] = self.get_latest_model_version(
        ).convertToDict()

        latest_version_dict["patch"] += 1

        return Model_Version(latest_version_dict)
    
    
    def _get_model_response(self, model: any, user_input: str, words: List[str], classes: List[str]) -> List[str]:
        bag_of_words: ndarray = self._get_word_bag(user_input, words)
        result: ndarray = model.predict(np.array([bag_of_words]))[0]
        threshold: float = 0.2
        result_list: List[List[float]] = [[index, res] for index, res in enumerate(result) if res > threshold]

        result_list.sort(key=lambda x: x[1], reverse=True)
        updated_result_list: List[float] = []

        for result in result_list:
            updated_result_list.append(classes[result[0]])

        return updated_result_list


    def _get_word_bag(self, user_input: str, words: List[str]) -> ndarray:
        tokens: List[str] = self._tokenize_input(user_input)
        bag_of_words: List[int] = [0] * len(words)

        for w in tokens:
            for index, word in enumerate(words):
                if word == w:
                    bag_of_words[index] = 1

        return np.array(bag_of_words)


    def _tokenize_input(self, user_input: str) -> List[str]:
        LEMMATIZER = WordNetLemmatizer()

        tokens: List[str] = word_tokenize(user_input)
        tokens: List[str] = [LEMMATIZER.lemmatize(token) for token in tokens]

        return tokens
    
    
    def _get_result(self, response: List[str], intents: Dict[str, List[Dict[str, any]]]) -> Dict[str, any]:
        tag: str = response[0]

        list_of_intents: List[Dict[str, any]] = intents["intents"]
        
        for intent in list_of_intents:
            if intent["tag"] == tag:
                return intent
            
        return "NO INTENT FOUND"