from kato_chat_bot.model.model_version import Model_Version
from typing import Dict

import pytest


@pytest.fixture
def fixture_version_dict() -> Dict[str, int]:
    return {
        "major": 1,
        "minor": 0,
        "patch": 0
    }


@pytest.fixture
def fixture_version_string() -> str:
    return "1_0_0"


def test_model_version_init_from_integers(fixture_version_dict: Dict[str, int]):
    model_version: Model_Version = Model_Version(fixture_version_dict)

    assert model_version.major == 1
    assert model_version.minor == 0
    assert model_version.patch == 0


def test_model_version_init_from_string(fixture_version_string: str):
    model_version: Model_Version = Model_Version(fixture_version_string)

    assert model_version.major == 1
    assert model_version.minor == 0
    assert model_version.patch == 0


def test_initialize_from_dict_error_one(fixture_version_dict: Dict[str, int]):
    del fixture_version_dict["major"]

    with pytest.raises(ValueError, match="Version dictionary should have major, minor, and patch as keys"):
        model_version: Model_Version = Model_Version(fixture_version_dict)


def test_initialize_from_dict_error_two(fixture_version_dict: Dict[str, int]):
    del fixture_version_dict["minor"]

    with pytest.raises(ValueError, match="Version dictionary should have major, minor, and patch as keys"):
        model_version: Model_Version = Model_Version(fixture_version_dict)


def test_initialize_from_dict_error_three(fixture_version_dict: Dict[str, int]):
    del fixture_version_dict["patch"]

    with pytest.raises(ValueError, match="Version dictionary should have major, minor, and patch as keys"):
        model_version: Model_Version = Model_Version(fixture_version_dict)


def test_initialize_from_string_wrong_length_error_one():
    fixture_version_string: str = "1_0"

    with pytest.raises(ValueError, match="Version string should be in the form x_y_z ex: 1_0_0"):
        model_version: Model_Version = Model_Version(fixture_version_string)


def test_initialize_from_string_wrong_length_error_two():
    fixture_version_string: str = "1_0_0_0"

    with pytest.raises(ValueError, match="Version string should be in the form x_y_z ex: 1_0_0"):
        model_version: Model_Version = Model_Version(fixture_version_string)


def test_initialize_from_string_not_integer_one():
    fixture_version_string: str = "a_0_0"

    with pytest.raises(ValueError, match="Major should be an integer and not empty, please double check your version string"):
        model_version: Model_Version = Model_Version(fixture_version_string)


def test_initialize_from_string_not_integer_two():
    fixture_version_string: str = "1_a_0"

    with pytest.raises(ValueError, match="Minor should be an integer and not empty, please double check your version string"):
        model_version: Model_Version = Model_Version(fixture_version_string)


def test_initialize_from_string_not_integer_three():
    fixture_version_string: str = "1_0_a"

    with pytest.raises(ValueError, match="Patch should be an integer and not empty, please double check your version string"):
        model_version: Model_Version = Model_Version(fixture_version_string)

def test_initialize_from_neither_dict_or_string():
    with pytest.raises(TypeError, match="Model version must either be a string or dictionary"):
        model_version: Model_Version = Model_Version([])
