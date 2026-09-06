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


def test_login_failures_are_not_reported_as_session_expiry(page):
    """The api() wrapper must not hijack 401s from /api/auth/* - a wrong password is not
    an expired session, and saying so sent a user hunting for the wrong problem."""
    js = _script(page)
    assert "!p.startsWith('/api/auth/')" in js, "401 interception must exempt auth endpoints"


def test_account_creation_is_acknowledged(page):
    """The gate used to just vanish on success, which is indistinguishable from nothing
    happening."""
    js = _script(page)
    assert "gOk" in js and "created" in js


def test_sign_in_and_create_are_separate_visible_modes(page):
    assert 'id="mSignin"' in page and 'id="mCreate"' in page


def test_hidden_attribute_is_authoritative(page):
    """An author `display:` rule outranks the UA's [hidden]{display:none}. Without a global
    override the sign-in panel stayed on screen after a successful login."""
    assert "[hidden]{display:none !important}" in _style(page)


def test_media_urls_carry_the_session_token(page):
    """<img> cannot send an Authorization header, so crops/images must pass the token
    another way - otherwise they are either broken or anonymously readable."""
    js = _script(page)
    assert "function media(" in js
    assert "media('/api/crop/" in js and "media('/api/image/" in js


def test_muted_text_meets_contrast_minimums(page):
    """--fg3 carries the small explanatory notes, where legibility matters most."""
    css = _style(page)
    def lum(hx):
        hx = hx.lstrip("#")
        c = [int(hx[i:i+2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    import re as _re
    fg3 = _re.search(r"--fg3:\s*(#[0-9a-fA-F]{6})", css).group(1)
    surface = _re.search(r"--surface:\s*(#[0-9a-fA-F]{6})", css).group(1)
    la, lb = lum(fg3), lum(surface)
    ratio = (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    assert ratio >= 4.5, f"muted text {fg3} on {surface} is {ratio:.2f}:1, below WCAG AA"


def test_sign_out_is_reachable_from_the_top_bar(page):
    assert 'id="signout"' in page and 'id="userchip"' in page


def test_a_like_does_not_remove_the_card(page):
    """judge() removed the card for BOTH verdicts, so favouriting a face made it vanish -
    which reads as the app eating your favourite."""
    js = _script(page)
    m = re.search(r"async function judge\(id,kind\)\{(.*?)\n\}", js, re.S)
    assert m, "judge() not found"
    body = m.group(1)
    assert "if(kind==='dislike')" in body, "removal must be conditional on rejection"
    filt = re.search(r"RES=RES\.filter\(x=>x\.face_id!==id\)", body)
    assert filt and body.index("if(kind==='dislike')") < filt.start()


def test_judging_refreshes_the_taste_readout(page):
    """With the default 10s undo window the sidebar kept reading OFF while the model was
    plainly learning."""
    js = _script(page)
    body = re.search(r"async function judge\(id,kind\)\{(.*?)\n\}", js, re.S).group(1)
    assert "await taste()" in body


def test_the_rerank_is_announced(page):
    """The post-undo-window re-rank happened silently, so it looked like nothing changed."""
    assert "Ranking updated from your taste" in _script(page)


def test_light_is_the_default_theme(page):
    assert "localStorage.getItem('t') || 'light'" in _script(page)


def test_both_themes_meet_contrast_minimums(page):
    css = _style(page)
    def lum(hx):
        hx = hx.lstrip("#")
        c = [int(hx[i:i+2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    def ratio(a, b):
        la, lb = lum(a), lum(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    light = re.search(r":root\{(.*?)\n\}", css, re.S).group(1)
    dark = re.search(r"\[data-theme=dark\]\{(.*?)\n\}", css, re.S).group(1)
    for name, block in (("light", light), ("dark", dark)):
        fg3 = re.search(r"--fg3:\s*(#[0-9a-fA-F]{6})", block).group(1)
        surf = re.search(r"--surface:\s*(#[0-9a-fA-F]{6})", block).group(1)
        r = ratio(fg3, surf)
        assert r >= 4.5, f"{name} muted text {fg3} on {surf} is {r:.2f}:1"


def test_toggle_switches_are_clickable(page):
    """The skin <i> is position:absolute;inset:0 and follows the input in DOM order, so it
    painted over the invisible checkbox and swallowed every click. None of the toggles
    worked. Two things fix it: the wrapper must be a <label> (so the whole control is a hit
    target) and the skin must not receive pointer events."""
    assert '<span class="sw">' not in page, "switch wrapper must be a <label>, not a <span>"
    assert page.count('<label class="sw"') >= 6
    css = _style(page)
    sw = re.search(r"\.sw i\{([^}]*)\}", css)
    assert sw and "pointer-events:none" in sw.group(1), \
        "the switch skin must let clicks reach the checkbox"


def test_every_switch_label_points_at_its_input(page):
    for m in re.finditer(r'<label class="sw" for="([A-Za-z0-9_]+)">\s*<input type="checkbox" id="([A-Za-z0-9_]+)"', page):
        assert m.group(1) == m.group(2), f"label for={m.group(1)} does not match input id={m.group(2)}"


def test_sidebar_groups_are_collapsible(page):
    """One long undifferentiated column is what made the panel feel crowded."""
    assert page.count('<details class="sect"') >= 6
    assert "<summary>" in page


def test_applied_filters_are_surfaced(page):
    """Selections must be visible without opening every collapsed group."""
    assert 'id="applied"' in page and "function appliedChips(" in _script(page)


def test_model_status_panel_exists(page):
    js = _script(page)
    assert 'id="modelSect"' in page
    assert "async function modelStatus(" in js and "/api/models" in js


def test_core_form_styles_are_present(page):
    """A block replacement once removed .f, .num, .sw and the range rules wholesale, which
    would have rendered the whole panel as unstyled browser defaults. Cheap to assert."""
    css = _style(page)
    for rule in (".f{", ".num{", ".sw{", ".sw i{", "input[type=range]{",
                 "::-moz-range-thumb", "::-webkit-slider-thumb"):
        assert rule in css, f"missing core style: {rule}"
