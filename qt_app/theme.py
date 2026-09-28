"""
Shared color palette + stylesheet for the PySide6 GUI (Step 2 of the
README's roadmap). Deliberately reuses the SAME hex values as the Tkinter
app's self._palette (gui/main_window.py __init__) plus the header
gradient/brand language from the confirmed Design mockups, so the two GUIs
look like the same product while this migration is in progress and the
Tkinter version is still the fallback.

One fixed palette now -- a dark-mode toggle used to live here (a second
DARK_PALETTE dict plus set_dark()/set_light() to swap PALETTE's contents in
place), but MM asked for it to be removed entirely rather than just hidden,
so it's gone: PALETTE is just this one dict, plain and simple, and nothing
mutates it at runtime.
"""

PALETTE = {
    "BG": "#f5f6f8",
    "SURFACE": "#ffffff",
    "HEADER_BG": "#152238",
    "HEADER_BG_2": "#1d3f6e",
    "ACCENT": "#2f6fb0",
    "ACCENT_DARK": "#24557f",
    "ACCENT_LIGHT": "#e8f0fb",
    "SUCCESS": "#3f9d6f",
    "SUCCESS_DARK": "#347f5a",
    "DANGER": "#c0392b",
    "DANGER_DARK": "#992d22",
    "TEXT": "#1a1d23",
    "TEXT_SOFT": "#4b5563",
    "MUTED": "#666666",
    "BORDER": "#d7dbe0",
    "CARD_BG": "#e9edf1",
    "PROGRESS_TRACK": "#e2e2e2",
    "ACCENT_DISABLED": "#a7c3dd",
    "APP_BRAND": "BehavioralTracker",
    "APP_VERSION": "v1.10",
}


def build_stylesheet(p=PALETTE):
    """One QSS string applied to the whole QApplication. Object names
    (#headerBar, #stepLabelActive, etc.) are set by the widgets that use
    them so this stays a single source of truth for colors."""
    return f"""
    QMainWindow, QWidget#centralWidget {{ background: {p['BG']}; }}

    QWidget#headerBar {{
        background: {p['HEADER_BG']};
        border-bottom: 1px solid {p['HEADER_BG']};
    }}
    QLabel#brandBadge {{
        background: {p['ACCENT']}; color: white; font-weight: 700;
        font-size: 15px; border-radius: 6px;
    }}
    QLabel#brandTitle {{ color: white; font-size: 17px; font-weight: 800; }}
    QLabel#brandVersion {{
        color: #bcd4f5; background: rgba(255,255,255,0.12);
        border-radius: 9px; font-size: 10px; font-weight: 700; padding: 2px 8px;
    }}
    QLabel#brandSubtitle {{ color: #93a6c2; font-size: 10.5px; }}
    QPushButton#headerBtn {{
        background: {p['HEADER_BG']}; color: white; font-size: 11px; font-weight: 600;
        border: 1px solid #33507a; border-radius: 6px; padding: 6px 12px;
    }}
    QPushButton#headerBtn:hover {{ background: {p['ACCENT_DARK']}; }}
    QPushButton#resetAllBtn {{
        background: {p['DANGER']}; color: white; font-size: 11px; font-weight: 700;
        border: none; border-radius: 6px; padding: 6px 14px;
    }}
    QPushButton#resetAllBtn:hover {{ background: {p['DANGER_DARK']}; }}

    QWidget#modeRow {{ background: {p['BG']}; }}

    QWidget#analysisRow {{ background: {p['BG']}; border-bottom: 1px solid {p['BORDER']}; }}
    QPushButton#analysisCard {{
        font-size: 11.5px; font-weight: 700; border-radius: 6px; padding: 8px 14px;
        background: {p['CARD_BG']}; color: {p['TEXT']}; border: 1px solid {p['BORDER']};
    }}
    QPushButton#analysisCardActive {{
        font-size: 11.5px; font-weight: 700; border-radius: 6px; padding: 8px 14px;
        background: {p['ACCENT']}; color: white; border: 1px solid {p['ACCENT']};
    }}
    QLabel#analysisSubtitle {{ color: {p['MUTED']}; font-size: 9.5px; }}

    QLabel#stepLabel {{ color: {p['MUTED']}; font-size: 11px; font-weight: 700; padding: 3px 8px; }}
    QLabel#stepLabelActive {{ color: {p['ACCENT']}; font-size: 11px; font-weight: 700; padding: 3px 8px; }}

    QGroupBox {{
        font-weight: 700; font-size: 11px; border: 1px solid {p['BORDER']};
        border-radius: 8px; margin-top: 10px; padding-top: 12px; background: {p['SURFACE']};
        color: {p['TEXT']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {p['TEXT']};
    }}

    QPushButton#accentBtn {{
        background: {p['ACCENT']}; color: white; font-weight: 700; border: none;
        border-radius: 6px; padding: 7px 14px;
    }}
    QPushButton#accentBtn:hover {{ background: {p['ACCENT_DARK']}; }}
    QPushButton#toolBtn {{
        background: {p['SURFACE']}; color: {p['TEXT']}; border: 1px solid {p['BORDER']};
        border-radius: 6px; padding: 7px 12px; font-size: 11px;
    }}
    QPushButton#toolBtn:hover {{ background: {p['ACCENT_LIGHT']}; }}
    QPushButton#toolBtnActive {{
        background: {p['ACCENT']}; color: white; border: 1px solid {p['ACCENT']};
        border-radius: 6px; padding: 7px 12px; font-size: 11px; font-weight: 700;
    }}
    QPushButton#dangerBtn {{
        background: {p['DANGER']}; color: white; border: none; border-radius: 6px;
        padding: 4px 10px; font-size: 10.5px; font-weight: 700;
    }}
    QPushButton#dangerBtn:hover {{ background: {p['DANGER_DARK']}; }}
    QPushButton#resetSmallBtn {{
        background: transparent; color: {p['MUTED']}; border: 1px solid {p['BORDER']};
        border-radius: 5px; padding: 2px 8px; font-size: 9.5px;
    }}
    QPushButton#startBtn {{
        background: {p['ACCENT']}; color: white; font-weight: 800; font-size: 13px;
        border: none; border-radius: 7px; padding: 12px;
    }}
    QPushButton#startBtn:hover {{ background: {p['ACCENT_DARK']}; }}
    QPushButton#startBtn:disabled {{ background: {p['ACCENT_DISABLED']}; }}

    QProgressBar {{
        border: 1px solid {p['BORDER']}; border-radius: 5px; background: {p['PROGRESS_TRACK']};
        text-align: center; height: 14px; color: {p['TEXT']};
    }}
    QProgressBar::chunk {{ background: {p['SUCCESS']}; border-radius: 4px; }}

    QWidget#footerBar {{ background: {p['CARD_BG']}; border-top: 1px solid {p['BORDER']}; }}
    QLabel#footerText {{ color: {p['TEXT_SOFT']}; font-size: 9.5px; }}
    QLabel#footerVersion {{ color: {p['ACCENT']}; font-size: 9.5px; font-weight: 700; }}

    QWidget#previewCanvas {{ background: #141a22; border-radius: 6px; }}

    QWidget, QLabel, QLineEdit, QComboBox, QCheckBox, QRadioButton {{
        color: {p['TEXT']};
    }}
    QLineEdit, QComboBox {{
        background: {p['SURFACE']}; border: 1px solid {p['BORDER']}; border-radius: 5px;
        padding: 3px 6px;
    }}
    QLineEdit:focus, QComboBox:focus {{ border: 1px solid {p['ACCENT']}; }}
    QLineEdit:disabled, QComboBox:disabled {{ background: {p['CARD_BG']}; color: {p['MUTED']}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background: {p['SURFACE']}; border: 1px solid {p['BORDER']}; selection-background-color: {p['ACCENT_LIGHT']};
        selection-color: {p['TEXT']}; outline: none;
    }}

    QCheckBox, QRadioButton {{ spacing: 7px; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 15px; height: 15px; border: 1px solid {p['BORDER']}; background: {p['SURFACE']};
    }}
    QCheckBox::indicator {{ border-radius: 4px; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border: 1px solid {p['ACCENT']}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: {p['ACCENT']}; border: 1px solid {p['ACCENT']};
    }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ background: {p['CARD_BG']}; }}

    /* Flat, modern scrollbars -- the left/right setup panels and the
    results page scroll internally (see _scroll_column in setup_page.py/
    results_page.py), so these show up often enough to be worth styling
    rather than leaving the chunky OS-default look MM's "beautiful" ask
    was about. */
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {p['BORDER']}; border-radius: 5px; min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {p['ACCENT']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; border: none; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
    QScrollBar:horizontal {{
        background: transparent; height: 10px; margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {p['BORDER']}; border-radius: 5px; min-width: 24px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {p['ACCENT']}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; border: none; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

    QToolTip {{
        background: {p['HEADER_BG']}; color: white; border: 1px solid {p['ACCENT']};
        border-radius: 4px; padding: 4px 7px; font-size: 10px;
    }}

    QScrollArea, QAbstractScrollArea {{ background: {p['BG']}; }}

    /* Plain, unnamed QWidget containers used purely as layout holders
    (a QScrollArea's inner content widget, a page's body/bottom strip)
    never get a background from the rules above -- QScrollArea's own
    "background" only paints its frame, not its internal viewport, and
    an ordinary QWidget falls back to Qt's default palette otherwise.
    Every such container is explicitly object-named (setup_page.py/
    results_page.py) so it can be targeted here directly, the same way
    every other themed widget in this stylesheet is. */
    QWidget#pageBody, QWidget#scrollContent {{ background: {p['BG']}; }}
    """
