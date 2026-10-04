"""The model pages of the documentation carry Python that evaluates each closed form next to the library's price.
This runs every block, so the pages cannot drift from the library."""
import pathlib
import re
import warnings
import pytest

MODELS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "models"
PAGES = sorted(p.stem for p in MODELS.glob("*.rst") if p.stem != "index")


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_model_page_runs(page):
    namespace = {}
    exec((MODELS / "helpers.py").read_text(), namespace)
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)", (MODELS / f"{page}.rst").read_text())
    assert blocks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for block in blocks:
            exec("\n".join(line[4:] for line in block.split("\n")), namespace)
