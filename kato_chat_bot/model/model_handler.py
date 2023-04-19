
import heapq
import random
import numpy as np
import tensorflow as tf

from typing import List, Any, Dict, Callable, Set
from os import listdir
from os.path import join, abspath, dirname

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

from nltk import download, word_tokenize
from nltk.stem import WordNetLemmatizer

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


    def create_and_save_model(self, version_param: str, intents: dict, function_mapper_file: any) -> None:
        version: Model_Version = Model_Version(version_param)
        
        if self.model_version_exists(version):
            return 

        if not self._download_packages():
            return

        intent_data = self._process_intents(intents)

        classes, words, document = intent_data["classes"], intent_data["words"], intent_data["document"]
        
        model = self._create_model(classes, words, document)
        
        model.save(f"{MODEL_DIRECTORY}/{str(version)}/model")
        
        self._save_model_misc(intent_data["classes"], f'{MODEL_DIRECTORY}/{str(version)}/classes.txt')
        self._save_model_misc(intent_data["words"], f'{MODEL_DIRECTORY}/{str(version)}/words.txt')


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

