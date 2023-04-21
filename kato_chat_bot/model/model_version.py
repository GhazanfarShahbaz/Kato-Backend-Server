from typing import Dict, List 

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
