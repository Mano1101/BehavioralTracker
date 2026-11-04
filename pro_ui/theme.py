"""Light / dark appearance that follows the computer's setting (Windows, macOS, Linux).

There is no theme switch in the program: when the system is in dark mode the whole window is dark,
otherwise light -- and it changes at once when the system setting changes.

How: every style sheet in the program is written for the light look.  In dark mode each colour in it is
swapped for its dark partner (a text colour for a light text colour, a background for a dark background),
and the application palette is set to match.  setStyleSheet() is wrapped once at start-up, so style
sheets set later (dialogs, buttons that change colour...) follow the theme too.
"""
import os
import re

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication, QWidget

# text colours (CSS 'color') -> readable on a dark background
TEXT_DARK = {
    "#1f2a44": "#e6eaf2", "#1b2a4a": "#f1f4fa", "#0f172a": "#f1f5f9", "#000000": "#e6eaf2", "#334155": "#cbd5e1",
    "#475569": "#a9b4c6", "#64748b": "#97a3b6", "#94a3b8": "#8592a8", "#1f6feb": "#6aa5ff", "#b45309": "#fbbf24",
    "#15803d": "#4ade80", "#16a34a": "#4ade80", "#b91c1c": "#f87171", "#dc2626": "#f87171", "#d92d20": "#f87171",
    "#7c3aed": "#a78bfa", "#d97706": "#fbbf24", "#1d4ed8": "#7cb2ff", "#9d4e8f": "#dd94d1", "#1e3a8a": "#93c5fd",
    "#8b95a7": "#8b95a7", "#cbd5e1": "#cbd5e1", "#ffffff": "#ffffff", "#e2e8f0": "#e2e8f0",
}
# backgrounds, borders, grid lines -> dark partners (the blue accent and status colours stay)
BG_DARK = {
    "#ffffff": "#1a2131", "#f3f6fb": "#0f141e", "#f7f9fc": "#1b2231", "#f6f9ff": "#1f2739", "#eef3fb": "#243046",
    "#eaf2ff": "#1f3252", "#e8f1ff": "#1c2c48", "#efe9ff": "#2a2346", "#e6f7ec": "#173222", "#fff0e0": "#3a2a16",
    "#fff5f4": "#3a1d1d", "#e9edf3": "#262e3e", "#e6ecf5": "#283143", "#dbe3f0": "#334058", "#dbe3ef": "#334058",
    "#e3e8f0": "#2b3445", "#eef2f8": "#242c3c", "#d5dce8": "#3b4659", "#c7d0de": "#48546b", "#cbd5e1": "#4b576d",
    "#e2e8f0": "#323c50", "#f0b4af": "#7f3a35", "#d5e2f5": "#2c3e5c", "#c8cdd8": "#4a5468", "#b6bfcc": "#56617a",
    "#1f2a44": "#2b3550", "#000000": "#000000",
}
NAMED = {"white": "#ffffff", "black": "#000000"}
TEXT_PROPS = {"color", "selection-color"}

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
_DECL = re.compile(r"([a-zA-Z-]+)(\s*:\s*)([^;{}]+)")

_state = {"dark": False, "installed": False, "orig_set": None, "watcher": None}


def _norm(c):
    c = c.lower()
    if len(c) == 4:
        c = "#" + "".join(ch * 2 for ch in c[1:])
    return c


def is_dark():
    return bool(_state["dark"])


def system_dark(app=None):
    """True when the operating system is in dark mode (BT_THEME=dark|light overrides it, for tests)."""
    env = os.environ.get("BT_THEME", "").lower()
    if env in ("dark", "light"):
        return env == "dark"
    app = app or QGuiApplication.instance()
    if app is None:
        return False
    try:
        cs = app.styleHints().colorScheme()
        if cs == Qt.ColorScheme.Dark:
            return True
        if cs == Qt.ColorScheme.Light:
            return False
    except Exception:
        pass
    return app.palette().color(QPalette.Window).lightness() < 110


def text(c):
    """A text colour for the current theme."""
    if not _state["dark"]:
        return c
    return TEXT_DARK.get(_norm(c), c)


def bg(c):
    """A background colour for the current theme."""
    if not _state["dark"]:
        return c
    return BG_DARK.get(_norm(c), c)


def themed(css):
    """A style sheet written for the light look, adapted to the current theme."""
    if not css or not _state["dark"]:
        return css

    def decl(m):
        prop, sep, val = m.group(1), m.group(2), m.group(3)
        table = TEXT_DARK if prop.lower() in TEXT_PROPS else BG_DARK
        val = _HEX.sub(lambda h: table.get(_norm(h.group(0)), h.group(0)), val)
        for word, hexc in NAMED.items():
            val = re.sub(rf"\b{word}\b", table.get(hexc, hexc), val)
        return prop + sep + val
    return _DECL.sub(decl, css)


def themed_html(html):
    """Inline colours in rich text (labels, text browsers)."""
    if not html or not _state["dark"]:
        return html
    return re.sub(r"(color\s*:\s*)(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})",
                  lambda m: m.group(1) + TEXT_DARK.get(_norm(m.group(2)), m.group(2)), html)


def palette(dark):
    p = QPalette()
    if dark:
        c = {QPalette.Window: "#0f141e", QPalette.WindowText: "#e6eaf2", QPalette.Base: "#1a2131",
             QPalette.AlternateBase: "#20283a", QPalette.Text: "#e6eaf2", QPalette.Button: "#1f2738",
             QPalette.ButtonText: "#e6eaf2", QPalette.BrightText: "#ffffff", QPalette.Highlight: "#1f6feb",
             QPalette.HighlightedText: "#ffffff", QPalette.ToolTipBase: "#2b3550", QPalette.ToolTipText: "#ffffff",
             QPalette.PlaceholderText: "#7d8aa0", QPalette.Link: "#6aa5ff", QPalette.Light: "#3a4559",
             QPalette.Midlight: "#2b3445", QPalette.Mid: "#232b3b", QPalette.Dark: "#0b0f17",
             QPalette.Shadow: "#000000"}
        dis = "#6b768a"
    else:
        c = {QPalette.Window: "#f3f6fb", QPalette.WindowText: "#1f2a44", QPalette.Base: "#ffffff",
             QPalette.AlternateBase: "#f7f9fc", QPalette.Text: "#1f2a44", QPalette.Button: "#ffffff",
             QPalette.ButtonText: "#1f2a44", QPalette.BrightText: "#ffffff", QPalette.Highlight: "#1f6feb",
             QPalette.HighlightedText: "#ffffff", QPalette.ToolTipBase: "#1f2a44", QPalette.ToolTipText: "#ffffff",
             QPalette.PlaceholderText: "#94a3b8", QPalette.Link: "#1f6feb", QPalette.Light: "#ffffff",
             QPalette.Midlight: "#eef2f8", QPalette.Mid: "#cbd5e1", QPalette.Dark: "#94a3b8",
             QPalette.Shadow: "#64748b"}
        dis = "#9aa5b5"
    for role, col in c.items():
        p.setColor(QPalette.All, role, QColor(col))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(dis))
    return p


def _wrapped_set(self, css):
    css = css or ""
    try:
        self.setProperty("_bt_ss", css)
    except Exception:
        pass
    _state["orig_set"](self, themed(css))


def _retext(w):
    """Rich-text labels with inline colours (also those built from the .ui file)."""
    from PySide6.QtWidgets import QLabel
    if not isinstance(w, QLabel) or _state.get("orig_label") is None:
        return
    src = w.property("_bt_txt")
    if src is None:
        t = w.text()
        if t and "color" in t and ("<" in t):
            w.setProperty("_bt_txt", t)
            src = t
    if src:
        _state["orig_label"](w, themed_html(src))


def _adopt(w):
    """A widget whose style sheet was set without the wrapper (e.g. loaded from the .ui file)."""
    if w.property("_bt_ss") is None:
        ss = w.styleSheet()
        if ss:
            w.setProperty("_bt_ss", ss)


def apply(app=None):
    """(Re)apply the theme that matches the system to the palette and every style sheet."""
    app = app or QApplication.instance()
    if app is None:
        return
    _state["dark"] = system_dark(app)
    app.setPalette(palette(_state["dark"]))
    orig = _state["orig_set"]
    if orig is None:
        return
    for w in app.allWidgets():
        _adopt(w)
        src = w.property("_bt_ss")
        if src:
            orig(w, themed(src))
        _retext(w)
    for w in app.topLevelWidgets():
        w.update()


class _Watcher(QObject):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.callbacks = []
        try:
            app.styleHints().colorSchemeChanged.connect(self._changed)
        except Exception:
            pass

    def _changed(self, *_a):
        was = _state["dark"]
        apply(self.app)
        if was != _state["dark"]:
            for cb in list(self.callbacks):
                try:
                    cb(_state["dark"])
                except Exception:
                    pass


def install(app=None):
    """Once, before any window is built: follow the system theme from now on."""
    app = app or QApplication.instance()
    if app is None:
        return
    if not _state["installed"]:
        _state["orig_set"] = QWidget.setStyleSheet
        QWidget.setStyleSheet = _wrapped_set
        # rich text with inline colours (status lines, info panels)
        from PySide6.QtWidgets import QLabel, QTextEdit
        orig_label = QLabel.setText
        orig_html = QTextEdit.setHtml

        def label_text(self, txt):
            rich = bool(txt) and "color" in txt
            try:
                if rich or self.property("_bt_txt") is not None:
                    self.setProperty("_bt_txt", txt if rich else None)
            except Exception:
                pass
            orig_label(self, themed_html(txt) if (_state["dark"] and rich) else txt)

        def text_html(self, html):
            orig_html(self, themed_html(html) if (_state["dark"] and html) else html)
        QLabel.setText = label_text
        QTextEdit.setHtml = text_html
        _state["orig_label"] = orig_label
        _state["watcher"] = _Watcher(app)
        _state["installed"] = True
    apply(app)


def on_change(cb):
    """cb(dark) is called when the system switches between light and dark."""
    if _state["watcher"] is not None:
        _state["watcher"].callbacks.append(cb)


def adopt_tree(root):
    """Re-theme a window built from a .ui file (its style sheets bypassed the wrapper)."""
    orig = _state["orig_set"]
    if orig is None or root is None:
        return
    for w in [root] + root.findChildren(QWidget):
        _adopt(w)
        src = w.property("_bt_ss")
        if src and _state["dark"]:
            orig(w, themed(src))
        _retext(w)
