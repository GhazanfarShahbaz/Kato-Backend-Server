from kato_chat_bot.model.model import Model_Version
from typing import Dict

import pytest


@pytest.fixture
def version_dict() -> Dict[str, int]:
    return {
        "major": 1,
        "minor": 0,
        "patch": 0
    }


@pytest.fixture
def version_string() -> str:
    return "1_0_0"


def test_model_version_init_from_integers(version_dict: Dict[str, int]):
    model_version: Model_Version = Model_Version(version_dict)

    assert model_version.major == 1
    assert model_version.minor == 0
    assert model_version.patch == 0


def test_model_version_init_from_string(version_string: str):
    model_version: Model_Version = Model_Version(version_string)

    assert model_version.major == 1
    assert model_version.minor == 0
    assert model_version.patch == 0


def test_initialize_from_dict_error_one(version_dict: Dict[str, int]):
    del version_dict["major"]

    with pytest.raises(ValueError):
        model_version: Model_Version = Model_Version(version_dict)


def test_initialize_from_dict_error_two(version_dict: Dict[str, int]):
    del version_dict["minor"]

    with pytest.raises(ValueError):
        model_version: Model_Version = Model_Version(version_dict)


def test_initialize_from_dict_error_three(version_dict: Dict[str, int]):
    del version_dict["patch"]

    with pytest.raises(ValueError):
        model_version: Model_Version = Model_Version(version_dict)


def test_initialize_from_neither_dict_or_string():
    with pytest.raises(TypeError):
        model_version: Model_Version = Model_Version([])
