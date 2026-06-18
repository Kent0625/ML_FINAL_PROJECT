from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_app_renders_without_streamlit_exceptions():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=20)

    assert len(app.exception) == 0
    assert len(app.button) == 1
    assert app.button[0].label == "Run historical pattern model"
    assert len(app.warning) == 1

    app.button[0].click().run(timeout=20)

    assert len(app.exception) == 0
    assert any(
        "Highest model probability" in markdown.value
        for markdown in app.markdown
    )
