"""The model pages of the documentation carry Python that evaluates each closed form next to the library's price,
followed by the output it printed. This runs every block and compares what it prints now with what the page shows,
so the pages cannot drift from the library."""
import contextlib
import io
import pathlib
import re
import warnings
import pytest

MODELS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "models"
PAGES = sorted(p.stem for p in MODELS.glob("*.rst") if p.stem != "index")
BLOCK = re.compile(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+?)\n*\.\. code-block:: text\n\n((?:    .*\n|\n)+)")
NUMBER = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def same(shown, printed):
    """The text matches and the numbers agree to the digits a page can promise across platforms. Numbers below 1e-6
    are error magnitudes at the level of a solver tolerance and are compared in order of magnitude only."""
    if NUMBER.sub("#", shown).split() != NUMBER.sub("#", printed).split():
        return False
    for a, b in zip(NUMBER.findall(shown), NUMBER.findall(printed)):
        a, b = float(a), float(b)
        if max(abs(a), abs(b)) < 1e-6:
            if (a == 0) != (b == 0) or (a and abs(a / b) > 100 or a and abs(b / a) > 100):
                return False
        elif abs(a - b) > 2e-5 * max(abs(a), abs(b)) + 1e-7:
            return False
    return True


def test_comparison_notices_a_changed_number():
    assert same("price  0.8343376435   error 3.7e-08", "price  0.8343376441   error 1.2e-08")
    assert not same("price  0.8343376435", "price  9.9999999999")
    assert not same("price  0.8343376435", "value  0.8343376435")


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_model_page_runs_and_shows_what_it_prints(page):
    namespace = {}
    exec((MODELS / "helpers.py").read_text(), namespace)
    blocks = BLOCK.findall((MODELS / f"{page}.rst").read_text())
    assert blocks
    for code, shown in blocks:
        printed = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(printed):
            warnings.simplefilter("ignore")
            exec("\n".join(line[4:] for line in code.split("\n")), namespace)
        shown = "\n".join(line[4:] for line in shown.rstrip().split("\n"))
        assert same(shown, printed.getvalue().rstrip()), f"{page}: the page shows\n{shown}\nbut the code prints\n{printed.getvalue()}"
