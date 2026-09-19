"""
=============================================================
  tabs/tab_clustering.py
  Onglet II – Clustering K-Means
  Affichage 2D et 3D avec score silhouette
=============================================================
"""

import tkinter as tk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from theme import COLORS, FONTS, TAB_COLORS
from utils.widgets import (Card, SectionTitle, MinMaxControl,
                            ActionButton, MetricBox, Separator, DisplayToggle, FieldLabel,ScrollableSidebar)
from utils.data_generator import generate_dataset_clustering

# Palette de couleurs pour les clusters
CLUSTER_COLORS = ["#A8C5DA", "#B8D4B8", "#D4B8D4", "#F4D4A8", "#D4A8A8", "#A8D4D4"]


class ClusteringTab(tk.Frame):
    def __init__(self, parent, go_to_page2=None):  # ← Ajout du callback
        """
        go_to_page2: callback pour revenir à la page 2 (grille des modules)
        """
        super().__init__(parent, bg=COLORS["bg_main"])
        self.accent = TAB_COLORS["clustering"]
        self.go_to_page2 = go_to_page2  # ← Stocker le callback
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chart_area()

    def _build_sidebar(self):
        # 1. Utilisation du composant défilable
        self.sidebar_comp = ScrollableSidebar(self, width=280, bg=COLORS["bg_card"])
        self.sidebar_comp.grid(row=0, column=0, sticky="ns", padx=(16, 8), pady=16)
        
        # 2. On utilise le frame intérieur du scroll pour placer les widgets
        sb = self.sidebar_comp.scrollable_frame
        p = 14

        # ── Bouton BACK vers Page 2 (AJOUTÉ) ─────────────────────
        back_btn = tk.Button(
            sb,
            text="◀ BACK to Modules",
            font=FONTS["body"],
            fg=COLORS["text_white"],
            bg=COLORS["accent_clust"],
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

        FieldLabel(sb, "Input Variables (X)").pack(anchor="w", padx=p, pady=(8, 2))

        self.x1 = MinMaxControl(sb, "X1", -10, 10)
        self.x1.pack(anchor="w", padx=p, pady=2)
        self.x2 = MinMaxControl(sb, "X2", -5, 15)
        self.x2.pack(anchor="w", padx=p, pady=2)
        self.x3 = MinMaxControl(sb, "X3", 0, 20)
        self.x3.pack(anchor="w", padx=p, pady=2)

        Separator(sb).pack(fill="x", padx=p, pady=8)

        FieldLabel(sb, "Number of clusters K").pack(anchor="w", padx=p, pady=(0, 2))
        self.k_var = tk.IntVar(value=3)
        tk.Spinbox(sb, from_=2, to=8, textvariable=self.k_var, width=6,
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

        # --- SECTION RÉSULTATS (Maintenant visible via le scroll) ---
        SectionTitle(sb, "📊  Results", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(8, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)

        self.m_k        = MetricBox(sb, "Clusters",  accent=self.accent)
        self.m_silhouette = MetricBox(sb, "Silhouette", accent=self.accent)
        for m in (self.m_k, self.m_silhouette):
            m.pack(fill="x", padx=p, pady=3)

        SectionTitle(sb, "📌  Cluster centers", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(12, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)
        
        self.centers_label = tk.Label(sb, text="—", font=FONTS["mono"],
                                       fg=COLORS["text_secondary"],
                                       bg=COLORS["bg_card"], justify="left")
        self.centers_label.pack(anchor="w", padx=p, pady=(0, 16))

    def _go_back_to_page2(self):
        """Retourne à la Page 2 (grille des modules)"""
        if self.go_to_page2:
            self.go_to_page2()

    def _build_chart_area(self):
        self.chart_frame = Card(self)
        self.chart_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        tk.Label(self.chart_frame,
                 text="📊  Configure the parameters and click Run",
                 font=FONTS["subtitle"], fg=COLORS["text_secondary"],
                 bg=COLORS["bg_card"]).pack(expand=True)

    def _run(self):
        n = self.n_var.get()
        k = self.k_var.get()
        x1_min, x1_max = self.x1.get()
        x2_min, x2_max = self.x2.get()
        x3_min, x3_max = self.x3.get()
        mode = self.display_toggle.get()

        X1, X2, X3 = generate_dataset_clustering(n, x1_min, x1_max, x2_min, x2_max, x3_min, x3_max)
        X = np.column_stack([X1, X2, X3])

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        sil = silhouette_score(X, labels) if k > 1 else 0

        self.m_k.set(k)
        self.m_silhouette.set(round(sil, 4))

        txt = ""
        for i, c in enumerate(centers):
            txt += f"C{i+1}: [{round(c[0],2)}, {round(c[1],2)}, {round(c[2],2)}]\n"
        self.centers_label.config(text=txt.strip())

        self._draw_chart(X1, X2, X3, labels, centers, k, mode)

    def _draw_chart(self, X1, X2, X3, labels, centers, k, mode):
        for w in self.chart_frame.winfo_children():
            w.destroy()

        colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in labels]

        fig = plt.Figure(figsize=(9, 6), facecolor=COLORS["bg_card"])

        if mode == "2D":
            pairs = [("X1", "X2", X1, X2, 0, 1),
                     ("X1", "X3", X1, X3, 0, 2),
                     ("X2", "X3", X2, X3, 1, 2)]
            for idx, (xl, yl, xa, ya, ci, cj) in enumerate(pairs):
                ax = fig.add_subplot(1, 3, idx + 1)
                ax.set_facecolor("#FAFBFC")
                ax.scatter(xa, ya, c=colors, alpha=0.6, s=18, edgecolors="#aaa", linewidths=0.3)
                ax.scatter(centers[:, ci], centers[:, cj], marker="*",
                           color="#E74C3C", s=180, zorder=5, label="Centroïde")
                ax.set_xlabel(xl, fontsize=8, color="#555")
                ax.set_ylabel(yl, fontsize=8, color="#555")
                ax.set_title(f"{xl} vs {yl}", fontsize=9, fontweight="bold", color="#333")
                ax.tick_params(labelsize=7)
                ax.spines[["top", "right"]].set_visible(False)
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("#FAFBFC")
            ax.scatter(X1, X2, X3, c=colors, alpha=0.6, s=18, edgecolors="#aaa", linewidths=0.3)
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                       marker="*", color="#E74C3C", s=200, zorder=5, label="Centroïdes")
            ax.set_xlabel("X1", fontsize=8, color="#555")
            ax.set_ylabel("X2", fontsize=8, color="#555")
            ax.set_zlabel("X3", fontsize=8, color="#555")
            ax.set_title(f"Clustering 3D — K={k}", fontsize=10, fontweight="bold", color="#333")
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)

        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)