"""
Small QPainter-based charts for the Results page that the Tkinter app draws
directly on a tk.Canvas rather than through matplotlib (TrajectoryWidget
mirrors the per-mouse polyline drawing in _show_results_multi_mouse,
EthogramWidget mirrors the bout-rectangle timeline in
_show_results_behavior). Standard Tracking's own Trajectory/Heatmap/Zone
Occupancy views are pre-rendered PNGs from output/graphs.py (matplotlib,
unchanged by this migration) and are just displayed as images -- no chart
widget needed for those.
"""

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QWidget

MOUSE_COLORS = ["#c0392b", "#2f6fb0", "#27ae60"]

BEHAVIOR_COLORS = {
    "locomotion": "#9e9e9e", "rearing": "#8e44ad", "grooming": "#d35400",
    "immobile": "#2c3e50", "other": "#bdbdbd", "undetermined": "#e0e0e0",
}


class TrajectoryWidget(QWidget):
    """Per-mouse colored path, fit to the widget -- mirrors the manual
    canvas.create_line loop in TrackerApp._show_results_multi_mouse."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(340)
        self.mouse_paths = {}  # {mouse_id: [(x, y), ...]}
        self.setStyleSheet("background: #dddddd; border-radius: 4px;")

    def set_data(self, df):
        self.mouse_paths = {}
        if df is None or len(df) == 0 or "mouse_id" not in df.columns:
            self.update()
            return
        for mouse_id in sorted(df["mouse_id"].unique()):
            sub = df[df["mouse_id"] == mouse_id].dropna(subset=["x", "y"])
            self.mouse_paths[mouse_id] = list(zip(sub["x"], sub["y"]))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#dddddd"))

        all_x, all_y = [], []
        for pts in self.mouse_paths.values():
            for x, y in pts:
                all_x.append(x)
                all_y.append(y)
        if not all_x:
            painter.setPen(QColor("#777777"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No tracking data")
            painter.end()
            return

        pad = 20
        cw, ch = self.width(), self.height()
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        sx = (cw - 2 * pad) / max(1.0, (x_max - x_min))
        sy = (ch - 2 * pad) / max(1.0, (y_max - y_min))
        scale = min(sx, sy)

        for i, (mouse_id, pts) in enumerate(sorted(self.mouse_paths.items())):
            color = QColor(MOUSE_COLORS[i % len(MOUSE_COLORS)])
            painter.setPen(QPen(color, 1))
            mapped = [(pad + (x - x_min) * scale, pad + (y - y_min) * scale) for x, y in pts]
            for j in range(len(mapped) - 1):
                painter.drawLine(mapped[j][0], mapped[j][1], mapped[j + 1][0], mapped[j + 1][1])
        painter.end()


class EthogramWidget(QWidget):
    """Per-subject horizontal bout timeline + legend -- mirrors the manual
    canvas.create_rectangle loop in TrackerApp._show_results_behavior."""

    def __init__(self):
        super().__init__()
        self.bouts_df = None
        self.duration_s = 1.0
        self.subjects = []
        self.setStyleSheet("background: white; border: 1px solid #c9c9c9; border-radius: 4px;")

    def set_data(self, bouts_df, duration_s):
        self.bouts_df = bouts_df
        self.duration_s = duration_s if duration_s and duration_s > 0 else 1.0
        self.subjects = sorted(bouts_df["subject"].unique()) if bouts_df is not None and len(bouts_df) else []
        self.setMinimumHeight(max(90, 34 * max(1, len(self.subjects)) + 50))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        label_w = 70
        cw = self.width()
        track_w = max(10, cw - label_w - 20)

        for i, subj in enumerate(self.subjects):
            y0 = 10 + i * 34
            painter.setPen(QColor("#1a1a1a"))
            bold = QFont(font)
            bold.setBold(True)
            painter.setFont(bold)
            painter.drawText(8, y0 + 15, subj)
            painter.setFont(font)
            sub_bouts = self.bouts_df[self.bouts_df["subject"] == subj]
            for _, r in sub_bouts.iterrows():
                x0 = label_w + (r["start_s"] / self.duration_s) * track_w
                x1 = label_w + (r["stop_s"] / self.duration_s) * track_w
                color = QColor(BEHAVIOR_COLORS.get(r["behavior"], "#e0e0e0"))
                painter.fillRect(QRectF(x0, y0, max(x1 - x0, 1), 20), color)

        legend_y = 10 + max(1, len(self.subjects)) * 34 + 6
        lx = label_w
        for name in ("locomotion", "rearing", "grooming", "immobile"):
            painter.fillRect(QRectF(lx, legend_y, 12, 12), QColor(BEHAVIOR_COLORS[name]))
            painter.setPen(QColor("#1a1a1a"))
            painter.drawText(lx + 16, legend_y + 10, name)
            lx += 95
        painter.end()
