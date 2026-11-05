import pytest

from conftest import ART, ROOT

pytestmark = pytest.mark.skipif(not (ART / "run_summary.json").exists(), reason="artifacts not present")
PAGES = ["Overview", "Models & ensembles", "Segments", "Contact budget", "Drivers", "Customer lookup"]


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def test_app_default_page():
    at = _app().run()
    assert not at.exception
    assert at.title[0].value.startswith("Investor Propensity")


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _app().run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.header[0].value == page


def test_budget_slider_changes_result():
    at = _app().run()
    at.sidebar.radio[0].set_value("Contact budget").run()
    at.slider[0].set_value(10).run()
    first = float(at.dataframe[0].value["buyers reached"].iloc[0])
    at.slider[0].set_value(30).run()
    assert not at.exception
    assert float(at.dataframe[0].value["buyers reached"].iloc[0]) > first


def test_lookup_changes_with_customer():
    at = _app().run()
    at.sidebar.radio[0].set_value("Customer lookup").run()
    assert not at.exception
    ids = at.selectbox[1].options
    at.selectbox[1].set_value(ids[-1]).run()
    assert not at.exception
