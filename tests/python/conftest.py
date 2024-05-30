# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-module-docstring,unused-import
import pytest


def pytest_configure(config):
    """Register markers"""
    config.addinivalue_line(
        "markers", "unittest: unittests for modules, do not require GPU, usually run fast"
    )
    config.addinivalue_line("markers", "op_correctness: unittest for op corectness, requires GPU")
    config.addinivalue_line(
        "markers",
        (
            "engine: testing engine feature functionalities, requires model and GPU, "
            "note: for most request related tests, use endpoint test instead."
        ),
    )
    config.addinivalue_line(
        "markers",
        (
            "endpoint: sending requests to a global endpoint fixture(can be an rest or API), "
            "tests compatibilities of API behaviors"
        ),
    )
    config.addinivalue_line(
        "markers",
        "uncategorized: this test is not yet categorized, team should work to categorize it",
    )
