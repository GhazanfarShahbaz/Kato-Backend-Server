from typing import Dict, List 

class Model_Version:
    def __init__(self, version: dict or str):
        def initialize_from_dict():
            """
                Uses the version dictionary to instantiate the Model_Version object
            """
            
            if not "major" in version or not "minor" in version or not "patch" in version:
                raise ValueError(
                    "Version dictionary should have major, minor, and patch as keys")

            self.major = version["major"]
            self.minor = version["minor"]
            self.patch = version["patch"]

        def initialize_from_string():
            """
                Uses the version string to instantiate the Model_Version object.
            """
            
            # NOTE: Add support for other formats    
            version_split: List[str] = version.split("_")
            version_length: int = len(version_split)

            # if the version length is not 3 then there are either not enough params or too much
            if version_length != 3:
                raise ValueError(
                    f"Version string should be in the form x_y_z ex: 1_0_0. Given: {version}")

            version_pointers: List[str] = ["Major", "Minor", "Patch"]

            for index, version_number in enumerate(version_split):
                version_type = version_pointers[index]

                try:
                    # Convert the version string to an integer
                    version_split[index] = int(version_number)
                except:
                    raise ValueError(
                        f"{version_type} should be an integer and not empty, please double check your version string. Given: {version}")

            self.major = version_split[0]
            self.minor = version_split[1]
            self.patch = version_split[2]

        # check if the version is either a dict or string, if not we cannot convert to Model_Version
        if isinstance(version, dict):
            initialize_from_dict()
        elif isinstance(version, str):
            initialize_from_string()
        else:
            raise TypeError(
                f"Model version must either be a string or dictionary not {type(version)}")

    def convertToDict(self) -> Dict[str, int]:
        """
            Convert model verstion to a dictionary
        """
        
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch
        }

    def __str__(self) -> str:
        """
            Convert model version to a string representation
        """
        return f"{self.major}_{self.minor}_{self.patch}"

    # https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate
    def __lt__(self, other_version) -> bool:
        # inverted for heap
        if self.major > other_version.major:
            return True
        elif self.major == other_version.major:
            if self.minor > other_version.minor:
                return True
            elif self.minor == other_version.minor and self.patch > other_version.patch:
                return True

        return False

    def __eq__(self, other_version) -> bool:
        return self.major == other_version.major and self.minor == other_version.minor and self.patch == other_version.patch
