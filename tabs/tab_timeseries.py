"""
=============================================================
  tabs/tab_timeseries.py
  Onglet IV – Séries Temporelles (ARIMA simplifié)
  Affichage 2D et 3D
=============================================================
"""

import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from theme import COLORS, FONTS, TAB_COLORS
from utils.widgets import (Card, SectionTitle, MinMaxControl,
                            ActionButton, MetricBox, Separator, DisplayToggle, FieldLabel, ScrollableSidebar)
from utils.data_generator import generate_timeseries


class TimeSeriesTab(tk.Frame):
    def __init__(self, parent, go_to_page2=None):  # ← Ajout du callback
        """
        go_to_page2: callback pour revenir à la page 2 (grille des modules)
        """
        super().__init__(parent, bg=COLORS["bg_main"])
        self.accent = TAB_COLORS["timeseries"]
        self.go_to_page2 = go_to_page2  # ← Stocker le callback
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chart_area()

    def _build_sidebar(self):
        # 1. Utilisation du composant défilable pour la sidebar
        self.sidebar_comp = ScrollableSidebar(self, width=280, bg=COLORS["bg_card"])
        self.sidebar_comp.grid(row=0, column=0, sticky="ns", padx=(16, 8), pady=16)
        
        # 2. On définit 'sb' comme le frame intérieur qui peut défiler
        sb = self.sidebar_comp.scrollable_frame
        p = 14

        # ── Bouton BACK vers Page 2 (AJOUTÉ) ─────────────────────
        back_btn = tk.Button(
            sb,
            text="◀ BACK to Modules",
            font=FONTS["body"],
            fg=COLORS["text_white"],
            bg=COLORS["accent_ts"],
            activebackground=COLORS["accent_clust"],
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._go_back_to_page2  # Action pour revenir à Page 2
        )
        back_btn.pack(anchor="w", padx=p, pady=(8, 8))

        SectionTitle(sb, "⚙️  Parameters", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(16, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)

        # Bloc des paramètres p, d, q
        for label, var_name, default in [("p (AR)", "p_var", 1), ("d (Diff.)", "d_var", 1), ("q (MA)", "q_var", 1)]:
            row = tk.Frame(sb, bg=COLORS["bg_card"])
            row.pack(anchor="w", padx=p, pady=2)
            tk.Label(row, text=label, font=FONTS["body_small"], fg=COLORS["text_secondary"],
                     bg=COLORS["bg_card"], width=10, anchor="w").pack(side="left")
            var = tk.IntVar(value=default)
            setattr(self, var_name, var)
            tk.Spinbox(row, from_=0, to=5, textvariable=var, width=4,
                       font=FONTS["body_small"], relief="flat", bg=COLORS["bg_main"],
                       highlightbackground=COLORS["border"], highlightthickness=1
                       ).pack(side="left", padx=4)

        Separator(sb).pack(fill="x", padx=p, pady=8)

        # Bouton bascule Affichage 2D / 3D
        self.display_toggle = DisplayToggle(sb)
        self.display_toggle.pack(anchor="w", padx=p, pady=4)
        
        Separator(sb).pack(fill="x", padx=p, pady=8)

        # Bouton exécution
        ActionButton(sb, "▶  Run", self._run, accent=self.accent
                     ).pack(fill="x", padx=p, pady=(4, 16))

        # --- SECTION RÉSULTATS ---
        SectionTitle(sb, "📊  ARIMA Results", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(8, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)
        
        self.m_mse  = MetricBox(sb, "MSE",  accent=self.accent)
        self.m_rmse = MetricBox(sb, "RMSE", accent=self.accent)
        for m in (self.m_mse, self.m_rmse):
            m.pack(fill="x", padx=p, pady=3)
            
        self.model_label = tk.Label(sb, text="—", font=FONTS["mono"],
                                     fg=COLORS["text_secondary"], bg=COLORS["bg_card"])
        self.model_label.pack(anchor="w", padx=p, pady=(4, 16))

    def _go_back_to_page2(self):
        """Retourne à la Page 2 (grille des modules)"""
        if self.go_to_page2:
            self.go_to_page2()

    def _build_chart_area(self):
        self.chart_frame = Card(self)
        self.chart_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        tk.Label(self.chart_frame,
                 text="⏱  Configure the parameters and click Run",
                 font=FONTS["subtitle"], fg=COLORS["text_secondary"],
                 bg=COLORS["bg_card"]).pack(expand=True)

    def _run(self):
        # Paramètres d'interface supprimés : passés en valeurs fixes par défaut pour éviter les erreurs
        n = 200  
        fcast = 30  
        y_min, y_max = 0, 15  
        trend = True  
        seasonal = True  
        
        p, d, q = self.p_var.get(), self.d_var.get(), self.q_var.get()
        mode = self.display_toggle.get()

        t, series = generate_timeseries(n, y_min, y_max, trend, seasonal)

        # Séparation train/test
        split = int(n * 0.8)
        train, test = series[:split], series[split:]

        # Prévision naïve (moving average) — simulation ARIMA
        window = max(p + d + q, 3)
        forecast = []
        history = list(train)
        for _ in range(len(test) + fcast):
            pred = np.mean(history[-window:])
            forecast.append(pred)
            if len(history) < len(series):
                history.append(series[len(history)])

        pred_test  = np.array(forecast[:len(test)])
        pred_future = np.array(forecast[len(test):])

        mse  = round(float(np.mean((test - pred_test)**2)), 4)
        rmse = round(float(np.sqrt(mse)), 4)

        self.m_mse.set(mse)
        self.m_rmse.set(rmse)
        self.model_label.config(text=f"Modèle ARIMA({p},{d},{q})\nMSE = {mse}\nRMSE = {rmse}")

        self._draw_chart(t, series, split, pred_test, pred_future, fcast, mode)

    def _draw_chart(self, t, series, split, pred_test, pred_future, fcast, mode):
        for w in self.chart_frame.winfo_children():
            w.destroy()

        fig = plt.Figure(figsize=(9, 6), facecolor=COLORS["bg_card"])

        if mode == "2D":
            ax = fig.add_subplot(111)
            ax.set_facecolor("#FAFBFC")

            ax.plot(t, series, color=TAB_COLORS["regression"], linewidth=1.2,
                    alpha=0.9, label="Série complète")
            t_test = t[split:]
            ax.plot(t_test, pred_test, color=TAB_COLORS["timeseries"], linewidth=1.5,
                    linestyle="--", label="Prédictions (test)")

            t_future = np.arange(len(t), len(t) + fcast)
            ax.plot(t_future, pred_future, color=TAB_COLORS["neuralnet"], linewidth=1.5,
                    linestyle="-.", label="Prévisions futures")
            ax.fill_between(t_future, pred_future * 0.9, pred_future * 1.1,
                             alpha=0.15, color=TAB_COLORS["neuralnet"])

            ax.axvline(x=t[split], color="#CCC", linewidth=1.2, linestyle=":")
            ax.text(t[split] + 1, series.max() * 0.95, "Test", fontsize=8, color="#999")

            ax.set_xlabel("Temps", fontsize=9, color="#555")
            ax.set_ylabel("Valeur", fontsize=9, color="#555")
            ax.set_title("Série Temporelle & Prévisions ARIMA", fontsize=10, fontweight="bold", color="#333")
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("#FAFBFC")
            phase = np.sin(t * 2 * np.pi / 30)
            ax.plot(t, series, phase, color=TAB_COLORS["regression"], linewidth=1.2, alpha=0.8, label="Série")
            t_future = np.arange(len(t), len(t) + fcast)
            phase_f  = np.sin(t_future * 2 * np.pi / 30)
            ax.plot(t_future, pred_future, phase_f, color=TAB_COLORS["neuralnet"],
                    linewidth=1.5, linestyle="-.", alpha=0.9, label="Prévisions")
            ax.set_xlabel("Temps", fontsize=8, color="#555")
            ax.set_ylabel("Valeur", fontsize=8, color="#555")
            ax.set_zlabel("Phase", fontsize=8, color="#555")
            ax.set_title("Série Temporelle 3D", fontsize=10, fontweight="bold", color="#333")
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)

        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)