"""
=============================================================
  tabs/tab_random_forest.py
  Onglet III – Random Forest (Classification)
  Importance des variables + Matrice de confusion
=============================================================
"""

import tkinter as tk
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from theme import COLORS, FONTS, TAB_COLORS
from utils.widgets import (Card, SectionTitle, MinMaxControl,
                            ActionButton, MetricBox, Separator, DisplayToggle, FieldLabel,ScrollableSidebar)
from utils.data_generator import generate_dataset_clustering


class RandomForestTab(tk.Frame):
    def __init__(self, parent, go_to_page2=None):  # ← Ajout du callback
        """
        go_to_page2: callback pour revenir à la page 2 (grille des modules)
        """
        super().__init__(parent, bg=COLORS["bg_main"])
        self.accent = TAB_COLORS["randomforest"]
        self.go_to_page2 = go_to_page2  # ← Stocker le callback
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chart_area()

    def _build_sidebar(self):
        self.sidebar = ScrollableSidebar(self, width=280)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=(16, 8), pady=16)
        
        sb = self.sidebar.scrollable_frame
        p = 14

        # ── Bouton BACK vers Page 2 (AJOUTÉ) ─────────────────────
        back_btn = tk.Button(
            sb,
            text="◀ BACK to Modules",
            font=FONTS["body"],
            fg=COLORS["text_white"],
            bg=COLORS["accent_rf"],
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

        FieldLabel(sb, "Variables (X1, X2, X3)").pack(anchor="w", padx=p, pady=(8, 2))
        self.x1 = MinMaxControl(sb, "X1", -10, 10)
        self.x1.pack(anchor="w", padx=p, pady=2)
        self.x2 = MinMaxControl(sb, "X2", -5, 15)
        self.x2.pack(anchor="w", padx=p, pady=2)
        self.x3 = MinMaxControl(sb, "X3", 0, 20)
        self.x3.pack(anchor="w", padx=p, pady=2)

        Separator(sb).pack(fill="x", padx=p, pady=8)

        FieldLabel(sb, "Number of trees").pack(anchor="w", padx=p, pady=(0, 2))
        self.trees_var = tk.IntVar(value=100)
        tk.Spinbox(sb, from_=10, to=500, increment=10,
                   textvariable=self.trees_var, width=8, font=FONTS["body_small"],
                   relief="flat", bg=COLORS["bg_main"],
                   highlightbackground=COLORS["border"], highlightthickness=1
                   ).pack(anchor="w", padx=p, pady=2)

        FieldLabel(sb, "Number of classes").pack(anchor="w", padx=p, pady=(8, 2))
        self.classes_var = tk.IntVar(value=3)
        tk.Spinbox(sb, from_=2, to=6, textvariable=self.classes_var, width=6,
                   font=FONTS["body_small"], relief="flat", bg=COLORS["bg_main"],
                   highlightbackground=COLORS["border"], highlightthickness=1
                   ).pack(anchor="w", padx=p, pady=2)

        FieldLabel(sb, "Dataset size").pack(anchor="w", padx=p, pady=(8, 2))
        self.n_var = tk.IntVar(value=200)
        tk.Spinbox(sb, from_=50, to=2000, increment=50,
                   textvariable=self.n_var, width=8, font=FONTS["body_small"],
                   relief="flat", bg=COLORS["bg_main"],
                   highlightbackground=COLORS["border"], highlightthickness=1
                   ).pack(anchor="w", padx=p, pady=2)

        Separator(sb).pack(fill="x", padx=p, pady=8)
        self.display_toggle = DisplayToggle(sb)
        self.display_toggle.pack(anchor="w", padx=p, pady=4)
        Separator(sb).pack(fill="x", padx=p, pady=8)

        ActionButton(sb, "▶  Run", self._run, accent=self.accent
                     ).pack(fill="x", padx=p, pady=(4, 16))

        SectionTitle(sb, "📊  Metrics", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(8, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)

        self.m_acc  = MetricBox(sb, "Accuracy",  accent=self.accent)
        self.m_f1   = MetricBox(sb, "F1 Score",    accent=self.accent)
        for m in (self.m_acc, self.m_f1):
            m.pack(fill="x", padx=p, pady=3)

        SectionTitle(sb, "🌿  Variable Importance", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(12, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)
        self.imp_label = tk.Label(sb, text="—", font=FONTS["mono"],
                                   fg=COLORS["text_secondary"],
                                   bg=COLORS["bg_card"], justify="left")
        self.imp_label.pack(anchor="w", padx=p, pady=(0, 16))

    def _go_back_to_page2(self):
        """Retourne à la Page 2 (grille des modules)"""
        if self.go_to_page2:
            self.go_to_page2()

    def _build_chart_area(self):
        self.chart_frame = Card(self)
        self.chart_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)

    def _run(self):
        n     = self.n_var.get()
        trees = self.trees_var.get()
        n_cls = self.classes_var.get()
        mode  = self.display_toggle.get()
        x1_min, x1_max = self.x1.get()
        x2_min, x2_max = self.x2.get()
        x3_min, x3_max = self.x3.get()

        X1, X2, X3 = generate_dataset_clustering(n, x1_min, x1_max, x2_min, x2_max, x3_min, x3_max)
        X = np.column_stack([X1, X2, X3])

        # Labels basés sur quantiles
        Y = np.digitize(X1 + X2, np.linspace(X1.min()+X2.min(),
                                               X1.max()+X2.max(), n_cls+1)[1:-1])

        split = int(n * 0.8)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        rf = RandomForestClassifier(n_estimators=trees, random_state=42)
        rf.fit(X_train, Y_train)
        Y_pred = rf.predict(X_test)

        acc = round(accuracy_score(Y_test, Y_pred), 4)
        f1  = round(f1_score(Y_test, Y_pred, average="macro", zero_division=0), 4)
        imp = rf.feature_importances_

        self.m_acc.set(acc)
        self.m_f1.set(f1)
        txt = (f"X1: {round(imp[0]*100,2)}%\n"
               f"X2: {round(imp[1]*100,2)}%\n"
               f"X3: {round(imp[2]*100,2)}%")
        self.imp_label.config(text=txt)

        self._draw_chart(rf, imp, Y_test, Y_pred, X_test, n_cls, mode)

    def _draw_chart(self, rf, imp, Y_test, Y_pred, X_test, n_cls, mode):
        for w in self.chart_frame.winfo_children():
            w.destroy()

        fig = plt.Figure(figsize=(9, 6.5), facecolor=COLORS["bg_card"])
        pastel_bars = [TAB_COLORS["randomforest"], TAB_COLORS["clustering"], TAB_COLORS["timeseries"]]

        if mode == "2D":
            # On utilise GridSpec pour un contrôle total des tailles
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2]) 
            ax1 = fig.add_subplot(gs[0, 0]) # Importance (Haut Gauche)
            ax2 = fig.add_subplot(gs[0, 1]) # Confusion (Haut Droite)
            ax3 = fig.add_subplot(gs[1, :]) # Frontières (Bas, prend toute la largeur)
            
            plt.rcParams.update({'font.size': 8})
            # Importance
            ax1.set_facecolor("#FAFBFC")
            bars = ax1.bar(["X1", "X2", "X3"], imp, color=pastel_bars, width=0.6)
            ax1.set_title("Importance des Variables", fontsize=9, fontweight="bold")
            for b, v in zip(bars, imp):
                ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                         f"{v:.1%}", ha="center", fontsize=8)

            # Matrice de confusion
            cm = confusion_matrix(Y_test, Y_pred)
            im = ax2.imshow(cm, cmap="Blues", aspect="equal")
            ax2.set_title("Matrice de Confusion", fontsize=9, fontweight="bold")
            ax2.set_xticks(range(n_cls))
            ax2.set_yticks(range(n_cls))
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            # --- AX3 : Frontières de Décision ---
            x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
            y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            
            # On fixe X3 à sa valeur moyenne pour la visualisation 2D
            x3_mean = np.full(xx.ravel().shape, X_test[:, 2].mean())
            Z = rf.predict(np.c_[xx.ravel(), yy.ravel(), x3_mean])
            Z = Z.reshape(xx.shape)

            ax3.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
            ax3.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=15, edgecolor='k', cmap="viridis")
            ax3.set_title("Frontières de décision (X1, X2)", fontsize=9, fontweight="bold")
            fig.tight_layout(pad=1.2)
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            colors_map = [TAB_COLORS["randomforest"], TAB_COLORS["clustering"],
                          TAB_COLORS["timeseries"], TAB_COLORS["neuralnet"],
                          TAB_COLORS["crossval"], TAB_COLORS["regression"]]
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("#FAFBFC")
            for cls in np.unique(Y_test):
                mask = Y_pred == cls
                ax.scatter(X_test[mask, 0], X_test[mask, 1], X_test[mask, 2],
                           s=20, alpha=0.6, color=colors_map[cls % len(colors_map)],
                           edgecolors="#aaa", linewidths=0.3, label=f"Classe {cls}")
            ax.set_xlabel("X1", fontsize=8, color="#555")
            ax.set_ylabel("X2", fontsize=8, color="#555")
            ax.set_zlabel("X3", fontsize=8, color="#555")
            ax.set_title("Random Forest — Prédictions 3D", fontsize=10, fontweight="bold", color="#333")
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)