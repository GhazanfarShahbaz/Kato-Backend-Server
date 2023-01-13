from kato_chat_bot.model.model import Model_Handler, Model_Version
from typing import Dict, List, Callable
from unittest import mock
from unittest.mock import patch, MagicMock
from copy import copy

import pytest

import heapq


@pytest.fixture
def fixture_available_versions() -> List[Model_Version]:
    return [
        Model_Version({
            "major": 1,
            "minor": 0,
            "patch": 0
        }),
        Model_Version("1_0_1")
    ]


@pytest.fixture
def fixture_version_dict() -> Dict[str, int]:
    return {
        "major": 1,
        "minor": 1,
        "patch": 1
    }


@pytest.fixture
def fixture_model_file_name() -> str:
    return "model_1_0_0"


@patch.object(Model_Handler, "_get_available_model_list")
def test_initialize(fixture_available_versions):
    Model_Handler._get_available_model_list.return_value = fixture_available_versions
    model_handler = Model_Handler()

    assert model_handler.available_versions == fixture_available_versions


@patch.object(Model_Handler, "_get_available_model_list")
def test_get_next_major_version(fixture_available_versions, fixture_version_dict):
    Model_Handler._get_available_model_list.return_value = fixture_available_versions
    model_handler = Model_Handler()

    Model_Handler.get_latest_model_version = MagicMock(
        return_value=Model_Version(fixture_version_dict), spec=dict)

    copy_version_dict = {
        "major": fixture_version_dict["major"] + 1,
        "minor": 0,
        "patch": 0
    }

    assert model_handler._get_next_major_version() == Model_Version(copy_version_dict)


@patch.object(Model_Handler, "_get_available_model_list")
def test_get_next_minor_version(fixture_available_versions, fixture_version_dict):
    Model_Handler._get_available_model_list.return_value = fixture_available_versions
    model_handler = Model_Handler()

    Model_Handler.get_latest_model_version = MagicMock(
        return_value=Model_Version(fixture_version_dict), spec=dict)

    copy_version_dict = {
        "major": fixture_version_dict["major"],
        "minor": fixture_version_dict["minor"] + 1,
        "patch": 0
    }

    assert model_handler._get_next_minor_version() == Model_Version(copy_version_dict)


@patch.object(Model_Handler, "_get_available_model_list")
def test_get_next_patch_version(fixture_available_versions, fixture_version_dict):
    Model_Handler._get_available_model_list.return_value = fixture_available_versions
    model_handler = Model_Handler()

    Model_Handler.get_latest_model_version = MagicMock(
        return_value=Model_Version(fixture_version_dict), spec=dict)

    copy_version_dict = {
        "major": fixture_version_dict["major"],
        "minor": fixture_version_dict["minor"],
        "patch": fixture_version_dict["patch"] + 1
    }

    assert model_handler._get_next_patch_version() == Model_Version(copy_version_dict)


@patch.object(Model_Handler, "_get_available_model_list")
def test_extract_model_version(fixture_available_versions, fixture_model_file_name):
    Model_Handler._get_available_model_list.return_value = fixture_available_versions
    model_handler = Model_Handler()

    assert model_handler._extract_model_version(
        fixture_model_file_name) == Model_Version("1_0_0")
