"""Custom Report builder dialog -- Upgrade Plan Tier 2 #10, opened from
ResultsPage's "Custom Report" button (see results_page.py's
_report_full_df/_report_id_columns/on_custom_report). Presents one
checkbox per available stat column (analysis.custom_report.
reportable_columns), ALL CHECKED BY DEFAULT so an untouched dialog just
reproduces today's "export everything" behavior, then writes the
narrowed table (analysis.custom_report.build_custom_report) to
Custom_Report.csv/.xlsx in the results folder.

Deliberately stateless across opens (no remembered selection) -- the
Upgrade Plan calls for "a simple" picker, and a fresh all-checked start
every time means there's no stale/mismatched selection to reconcile when
a different analysis type's (differently-shaped) results are shown."""

import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton,
    QScrollArea, QWidget, QMessageBox,
)

from analysis.custom_report import reportable_columns, build_custom_report


def open_report_builder_dialog(parent, full_df, id_columns, output_dir):
    dialog = QDialog(parent)
    dialog.setWindowTitle("Custom Report")
    dialog.setModal(True)
    dialog.resize(420, 520)
    layout = QVBoxLayout(dialog)

    intro = QLabel(
        "Pick which stats to include (all checked by default). Saves as "
        "Custom_Report.csv/.xlsx in the results folder."
    )
    intro.setWordWrap(True)
    intro.setStyleSheet("color: #666666; font-size: 9px;")
    layout.addWidget(intro)

    columns = reportable_columns(full_df, id_columns=id_columns)

    top_btns = QHBoxLayout()
    all_btn = QPushButton("Select All")
    none_btn = QPushButton("Select None")
    top_btns.addWidget(all_btn)
    top_btns.addWidget(none_btn)
    top_btns.addStretch()
    layout.addLayout(top_btns)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content = QWidget()
    clayout = QVBoxLayout(content)
    checks = {}
    if columns:
        for col in columns:
            cb = QCheckBox(col)
            cb.setChecked(True)
            clayout.addWidget(cb)
            checks[col] = cb
    else:
        clayout.addWidget(QLabel("No optional stat columns available for this run."))
    clayout.addStretch()
    scroll.setWidget(content)
    layout.addWidget(scroll, 1)

    all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checks.values()])
    none_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checks.values()])

    status_label = QLabel("")
    status_label.setWordWrap(True)
    status_label.setStyleSheet("color: #666666; font-size: 9px;")
    layout.addWidget(status_label)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.reject)
    save_btn = QPushButton("Save Report")
    save_btn.setObjectName("accentBtn")
    btn_row.addWidget(close_btn)
    btn_row.addWidget(save_btn)
    layout.addLayout(btn_row)

    def do_save():
        selected = [col for col, cb in checks.items() if cb.isChecked()]
        report_df = build_custom_report(full_df, selected, id_columns=id_columns)
        if len(report_df.columns) == 0:
            QMessageBox.warning(dialog, "Nothing to save", "Select at least one stat to include.")
            return
        csv_path = os.path.join(output_dir, "Custom_Report.csv")
        xlsx_path = os.path.join(output_dir, "Custom_Report.xlsx")
        try:
            os.makedirs(output_dir, exist_ok=True)
            report_df.to_csv(csv_path, index=False)
            report_df.to_excel(xlsx_path, index=False, sheet_name="Custom Report")
        except OSError as exc:
            QMessageBox.critical(dialog, "Could not save report", str(exc))
            return
        kept_stats = max(len(report_df.columns) - len(id_columns), 0)
        status_label.setText(f"Saved {kept_stats} stat column(s) -- {csv_path}")

    save_btn.clicked.connect(do_save)
    dialog.exec()
