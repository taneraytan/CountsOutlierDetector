"""End-to-end smoke tests for the Streamlit app via streamlit.testing.

These run the real app script in-process with an isolated database (the
COUNTS_DB_PATH env var is set in conftest.py before any import).
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "ui.py")


def _boot() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
    return at


def _goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"page '{page}' raised: {at.exception}"
    return at


def test_home_page_renders():
    at = _boot()
    assert len(at.sidebar.radio) == 1
    assert "Home" in at.sidebar.radio[0].options


@pytest.mark.parametrize("page", [
    "1 · Load Data",
    "2 · Preprocess",
    "3 · Feature Engineering",
    "4 · Configure & Run",
    "5 · Results",
    "History",
])
def test_every_page_renders_without_data(page):
    at = _boot()
    _goto(at, page)


def test_demo_load_and_full_run():
    at = _boot()
    _goto(at, "1 · Load Data")
    # The demo tab's "Load demo dataset" button.
    demo_buttons = [b for b in at.button if "demo" in str(b.label).lower()]
    assert demo_buttons, "demo button not found"
    demo_buttons[0].click().run()
    assert not at.exception

    _goto(at, "4 · Configure & Run")
    run_buttons = [b for b in at.button if str(b.label) == "Run analysis"]
    assert run_buttons, "Run analysis button not found"
    run_buttons[0].click().run(timeout=120)
    assert not at.exception

    _goto(at, "5 · Results")
    # Results should be populated: at least one altair chart and no error.
    assert not at.exception

    _goto(at, "History")
    assert not at.exception
