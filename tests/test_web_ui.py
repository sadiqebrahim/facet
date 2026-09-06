"""Static checks on the single-file web app.

These exist because of two bugs that shipped and broke the UI for a real user:

* `id="taste%"` — `%` is not valid in a bare `#id` CSS selector, so `querySelector` threw.
  It surfaced only as a red error under the login form, and the account had in fact been
  created. My earlier check compared JS-referenced ids against HTML ids, so `taste%` matched
  itself on both sides and passed.
* ~100 lines of JavaScript were patched **into the `<style>` element**, because the marker
  comment used to anchor the edit existed in both the CSS and the JS. The code was inert
  there and silently duplicated later in the script.

Neither is caught by a syntax check. These assertions are cheap and catch both.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "web" / "index.html"

pytestmark = pytest.mark.skipif(not HTML.exists(), reason="web UI not present")


@pytest.fixture(scope="module")
def page():
    return HTML.read_text()


def _style(page):
    return page[page.index("<style>") + 7 : page.index("</style>")]


def _script(page):
    return page[page.index("<script>") + 8 : page.rindex("</script>")]


def _ids(page):
    # ignore ids built from template literals at runtime
    return [i for i in re.findall(r'id="([^"]+)"', page) if "${" not in i]


def test_every_id_is_a_valid_css_selector(page):
    bad = [i for i in _ids(page) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", i)]
    assert not bad, f"ids unusable as a bare #selector: {bad}"


def test_no_javascript_leaked_into_the_stylesheet(page):
    css = _style(page)
    leaks = [ln for ln in css.splitlines()
             if re.match(r"^\s*(async function |function |let [A-Z]|const \w+ ?= ?\(|\$\()", ln)]
    assert not leaks, f"JavaScript inside <style>: {leaks[:3]}"


def test_no_css_leaked_into_the_script(page):
    js = _script(page)
    leaks = [ln for ln in js.splitlines()
             if re.match(r"^[.#][a-zA-Z][\w -]*\{", ln)]
    assert not leaks, f"CSS inside <script>: {leaks[:3]}"


def test_no_duplicate_function_definitions(page):
    names = re.findall(r"^(?:async )?function (\w+)", _script(page), re.M)
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"defined more than once: {sorted(dupes)}"


def test_css_braces_balanced(page):
    depth = 0
    for ch in _style(page):
        depth += ch == "{"
        depth -= ch == "}"
        assert depth >= 0, "unbalanced closing brace in CSS"
    assert depth == 0, f"unclosed CSS block (depth {depth})"


def test_every_referenced_element_exists(page):
    ids = set(_ids(page))
    refs = set(re.findall(r"\$\$?\('#([A-Za-z0-9_-]+)'\)", _script(page)))
    assert not (refs - ids), f"script queries ids that do not exist: {sorted(refs - ids)}"


def test_viewport_allows_zoom_and_handles_notches(page):
    assert "viewport-fit=cover" in page
    assert "user-scalable=no" not in page, "pinch-zoom must not be disabled"


def test_touch_devices_get_the_hover_only_controls(page):
    """The judge buttons and favourite star are revealed by :hover on desktop; without a
    hover:none override they are unreachable on a phone."""
    css = _style(page)
    block = re.search(r"@media \(hover:none\)\{(.*?)\n\}", css, re.S)
    assert block, "no @media (hover:none) block"
    assert ".judge" in block.group(1) and ".star" in block.group(1)


def test_range_inputs_are_styled_for_firefox(page):
    css = _style(page)
    assert "::-moz-range-thumb" in css, "Firefox renders a default slider without this"
    assert "::-webkit-slider-thumb" in css


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_script_parses(page, tmp_path):
    f = tmp_path / "ui.js"
    f.write_text(_script(page))
    r = subprocess.run([shutil.which("node"), "--check", str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
