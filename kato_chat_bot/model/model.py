
import heapq
import json
import random

from typing import List, Any, Dict, Callable, Set
from ctypes import cast, py_object
from os import listdir

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.models import load_model

# from nltk import download, word_tokenize
# from nltk.stem import WordNetLemmatizer

# from model_settings.settings import PACKAGES


MODEL_DIRECTORY: str = "saved_models"


class Model_Version:
    def __init__(self, version: dict or str):
        def initialize_from_dict():
            if not "major" in version or not "minor" in version or not "patch" in version:
                raise ValueError(
                    "Version dictionary should have major, minor, and patch as keys")

            self.major = version["major"]
            self.minor = version["minor"]
            self.patch = version["patch"]

        def initialize_from_string():
            version_split: List[str] = version.split("_")
            version_length: int = len(version_split)

            if version_length != 3:
                raise ValueError(
                    f"Version string should be in the form x_y_z ex: 1_0_0. Given: {version}")

            version_pointers = ["Major", "Minor", "Patch"]

            for index, version_number in enumerate(version_split):
                version_type = version_pointers[index]

                try:
                    version_split[index] = int(version_number)
                except:
                    raise ValueError(
                        f"{version_type} should be an integer and not empty, please double check your version string. Given: {version}")

            self.major = version_split[0]
            self.minor = version_split[1]
            self.patch = version_split[2]

        if isinstance(version, dict):
            initialize_from_dict()
        elif isinstance(version, str):
            initialize_from_string()
        else:
            raise TypeError(
                f"Model version must either be a string or dictionary not {type(version)}")

    def convertToDict(self) -> Dict[str, int]:
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch
        }

    def __str__(self) -> str:
        return f"{self.major}_{self.minor}_{self.patch}"

    # https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate
    def __lt__(self, other_version):
        # inverted for heap
        if self.major > other_version.major:
            return True
        elif self.major == other_version.major:
            if self.minor > other_version.minor:
                return True
            elif self.minor == other_version.minor and self.patch > other_version.patch:
                return True

        return False

    def __eq__(self, other_version):
        return self.major == other_version.major and self.minor == other_version.minor and self.patch == other_version.patch


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

#     def get_model(model_version: Model_Version) -> Any:
#         model_version_string: str = str(model_version)

#         return load_model(f"saved_model/model_v_{model_version_string}")

#     def model_version_exists(self, version: Model_Version) -> bool:
#         return version in self.available_versions


#     def create_and_save_model(self, version_param: str, intents: dict, patterns: dict, function_mapper_file) -> None:
#         version: Model_Version = self.model_generator_params[version_param]()

#         if not self._download_packages():
#             return

#         intent_data = self._process_intents(intents)

#         classes, words, document = intent_data["classes"], intent_data["words"], intent_data["document"]


#     # PRIVATE FUNCTIONS START HERE
#     def _download_packages() -> bool:
#         package_count: int = len(PACKAGES)

#         for i, package in enumerate(PACKAGES):
#             try:
#                 download(package, quiet=True)
#                 print(f"{i+1}/{package_count}: Downloaded {package}")
#             except:
#                 print(f"Could not download {package}")
#                 return False

#         print("Downloaded all packages")
#         return True

#     def _process_intents(intents: dict) -> dict:
#         lemmatizer = WordNetLemmatizer()

#         classes: Set[str] = set()
#         words: Set[str] = set()
#         doc_x, doc_y = [], []

#         for intent in intents["intents"]:
#             for pattern in intent["patterns"]:
#                 tokens: List[str] = word_tokenize(pattern) #extract words from each pattern
#                 tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

#                 words.update(tokens)

#                 doc_x.append(pattern)
#                 doc_y.append(intent["tag"])

#             classes.add(intent["tag"])

#         return {
#             "classes"   : sorted(classes),
#             "words"     : sorted(words),
#             "document"  : {
#                 "x": doc_x,
#                 "y": doc_y
#             }
#         }

#     def _create_model(classes: str, words: str, documents: Dict[str, List[str]]) -> any:
#         document_x, document_y = documents["x"], documents["y"]
#         lemmatizer = WordNetLemmatizer()

#         training_data = []
#         out_empty: List[int] = [0] * len(classes)

#         for index, document in enumerate(document_x):
#             bag_of_words = []
#             text = lemmatizer.lemmatize(document.lower())

#             for word in words:
#                 bag_of_words.append(1) if word in text else bag_of_words.append(0)

#             output_row = list(out_empty)
#             output_row[classes.index(document_y[index])] = 1
#             training_data.append([bag_of_words, output_row])


#             random.shuffle(training_data)
#             training_data = np.array(training_data, dtype = object)

#             x = np.array(list(training_data[:, 0])) # first training phase
#             y = np.array(list(training_data[:, 1])) # second training phase

#             i_shape = (len(x[0]), )
#             o_shape = len(y[0])

#             model = Sequential()
#             model.add(Dense(128, input_shape = i_shape, activation = "relu"))
#             model.add(Dropout(0.5))

#             model.add(Dense(64, activation = "relu"))
#             model.add(Dropout(0.3))
#             model.add(Dense(o_shape, activation = "softmax"))

#             md = tf.keras.optimizers.Adam(learning_rate = 0.01, decay = 1e-6)

#             model.compile(
#                 loss        = 'categorical_crossentropy',
#                 optimizer   = md,
#                 metrics     = ["accuracy"]
#             )

#             model.fit(x, y, epochs = 1000, verbose = 1)

#         return model

#     def _save_model_misc(data: any, file_name: str) -> None:
#         data_file = open(file_name, "w")

#         for element in data:
#             data_file.write(element + "\n")

#         data_file.close

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
