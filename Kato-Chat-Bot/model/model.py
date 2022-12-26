import os
import heapq

from typing import List, Any
from tensorflow.keras.models import load_model

MODEL_DIRECTORY: str = "saved_models"

class Model_Version:
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __init__(self, version_string: str):
        version_split: List[str] = version_string.split("_")

        self.major = int(version_split[0])
        self.minor = int(version_split[1])
        self.patch = int(version_split[2])

    def __str__(self) -> str:
        return f"{self.major}_{self.minor}_{self.patch}"

    # https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate
    def __lt__(self, other_version):
        if self.major < other_version.major:
            return True
        elif self.major == other_version.major:
            if self.minor < other_version.minor:
                return True
            elif self.minor == other_version.minor and self.patch < other_version.patch:
                return True

        return False


class Models:
    def __init__(self):
        self.available_versions: List[Model_Version] = self._get_available_model_list()

    def get_latest_model_version(self) -> Model_Version:
        latest_version: Model_Version = heapq.heappop(self.available_versions)
        heapq.heappush(self.available_versions, latest_version)

        return latest_version

    def get_model(model_version: Model_Version) -> Any:
        model_version_string: str = str(model_version)

        return load_model(f"saved_model/model_v_{model_version_string}")

    def _get_available_model_list(self) -> List[Model_Version]:
        model_versions: List[Model_Version] = []

        for file_name in os.listdir(MODEL_DIRECTORY):
            if file_name[:5] == "model":
                model_versions.append(self._extract_model_version(file_name))

        heapq._heapify_max(model_versions)

        return model_versions

    def _extract_model_version(file_name: str) -> Model_Version:
        version_string: str = file_name[9:]

        return Model_Version(version_string=version_string)
