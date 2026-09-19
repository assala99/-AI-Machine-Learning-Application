"""
=============================================================
  tabs/tab_regression.py
  Onglet I – Régression Linéaire Multiple
  Inputs : X1, X2 → Output : Y (généré automatiquement)
  Affichage 2D et 3D
=============================================================
"""

import tkinter as tk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa

from theme import COLORS, FONTS, TAB_COLORS
from utils.widgets import (Card, SectionTitle, MinMaxControl,
                            ActionButton, Separator, DisplayToggle, FieldLabel, ScrollableSidebar)


class RegressionTab(tk.Frame):
    def __init__(self, parent, go_to_page2=None):  # Ajout du callback
        """
        go_to_page2: callback pour revenir à la page 2 (grille des modules)
        """
        super().__init__(parent, bg=COLORS["bg_main"])
        self.accent = TAB_COLORS["regression"]
        self.go_to_page2 = go_to_page2  # Stocker le callback
        self._build_ui()

    # ── Construction de l'interface ──────────────────────────
    def _build_ui(self):
        # Layout principal : sidebar gauche + zone graphique droite
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_chart_area()

    def _build_sidebar(self):
        # Utilisation du composant défilable
        self.sidebar_comp = ScrollableSidebar(self, width=320, bg=COLORS["bg_card"])
        self.sidebar_comp.grid(row=0, column=0, sticky="ns", padx=(16, 8), pady=16)
        
        sb = self.sidebar_comp.scrollable_frame
        p = 14  # padding interne

        # ── Bouton BACK vers Page 2 (AJOUTÉ) ─────────────────────
        back_btn = tk.Button(
            sb,
            text="◀ BACK to Modules",
            font=FONTS["body"],
            fg=COLORS["text_white"],
            bg=COLORS["accent_reg"],
            activebackground=COLORS["accent_clust"],
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._go_back_to_page2  # Action pour revenir à Page 2
        )
        back_btn.pack(anchor="w", padx=p, pady=(8, 8))
    
        SectionTitle(sb, "Parameters", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(16, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)
    
        # ── Variables d'entrée (X1 et X2 seulement) ───────────────
        FieldLabel(sb, "Input Variables (X)").pack(anchor="w", padx=p, pady=(8, 2))
    
        self.x1 = MinMaxControl(sb, "X1", -10, 10)
        self.x1.pack(anchor="w", padx=p, pady=2)
    
        self.x2 = MinMaxControl(sb, "X2", -5, 15)
        self.x2.pack(anchor="w", padx=p, pady=2)
    
        Separator(sb).pack(fill="x", padx=p, pady=8)
    
        # ── Mode affichage ───────────────────────────────────
        FieldLabel(sb, "Display :").pack(anchor="w", padx=p, pady=(0, 2))
        self.display_toggle = DisplayToggle(sb)
        self.display_toggle.pack(anchor="w", padx=p, pady=4)
    
        Separator(sb).pack(fill="x", padx=p, pady=12)
    
        # ── Bouton exécution ─────────────────────────────────
        ActionButton(sb, "▶  Run", self._run, accent=self.accent
                     ).pack(fill="x", padx=p, pady=(4, 16))
    
        # ── Métriques ────────────────────────────────────────
        SectionTitle(sb, "Performance Methods:", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(8, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)
    
        # Frame pour les métriques
        self.metrics_frame = tk.Frame(sb, bg=COLORS["bg_card"])
        self.metrics_frame.pack(fill="x", padx=p, pady=8)
        
        self.metrics_label = tk.Label(self.metrics_frame, 
                                       text="—",
                                       font=FONTS["mono"],
                                       fg=COLORS["text_secondary"],
                                       bg=COLORS["bg_card"],
                                       justify="left")
        self.metrics_label.pack(anchor="w", padx=10, pady=8)
    
        # ── Coefficients ─────────────────────────────────────
        SectionTitle(sb, "Model Coefficients:", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(12, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)
        
        self.coeff_frame = tk.Frame(sb, bg=COLORS["bg_card"])
        self.coeff_frame.pack(fill="x", padx=p, pady=8)
        
        self.coeff_label = tk.Label(self.coeff_frame, 
                                     text="—",
                                     font=FONTS["mono"],
                                     fg=COLORS["text_secondary"],
                                     bg=COLORS["bg_card"],
                                     justify="left")
        self.coeff_label.pack(anchor="w", padx=10, pady=8)
        
        # ── Équation du modèle ───────────────────────────────
        SectionTitle(sb, "Model Equation:", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(12, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)
        
        self.equation_frame = tk.Frame(sb, bg=COLORS["bg_card"])
        self.equation_frame.pack(fill="x", padx=p, pady=8)
        
        self.equation_label = tk.Label(self.equation_frame, 
                                        text="—",
                                        font=FONTS["mono"],
                                        fg=COLORS["text_secondary"],
                                        bg=COLORS["bg_card"],
                                        justify="left",
                                        wraplength=280)
        self.equation_label.pack(anchor="w", padx=10, pady=8)

    def _go_back_to_page2(self):
        """Retourne à la Page 2 (grille des modules)"""
        if self.go_to_page2:
            self.go_to_page2()

    def _build_chart_area(self):
        self.chart_frame = Card(self)
        self.chart_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)

        # Placeholder
        tk.Label(self.chart_frame,
                 text="📈  Configure parameters and click Run",
                 font=FONTS["subtitle"], fg=COLORS["text_secondary"],
                 bg=COLORS["bg_card"]
                 ).pack(expand=True)

    # ── Génération des données (aléatoire à chaque exécution) ──
    def _generate_data(self, n_samples=200, x1_min=-10, x1_max=10, x2_min=-5, x2_max=15):
        """Génère des données aléatoires à chaque appel"""
        X1 = np.random.uniform(x1_min, x1_max, n_samples)
        X2 = np.random.uniform(x2_min, x2_max, n_samples)
        
        # Coefficients réels aléatoires
        true_intercept = np.random.uniform(-10, 20)
        true_coef1 = np.random.uniform(-5, 8)
        true_coef2 = np.random.uniform(-4, 7)
        noise = np.random.normal(0, np.random.uniform(2, 8), n_samples)
        
        Y = true_intercept + true_coef1 * X1 + true_coef2 * X2 + noise
        
        return X1, X2, Y

    # ── Logique ML ──────────────────────────────────────────
    def _run(self):
        # Récupération des paramètres
        mode = self.display_toggle.get()
        x1_min, x1_max = self.x1.get()
        x2_min, x2_max = self.x2.get()
        
        # Génération des données (aléatoire à chaque exécution)
        X1, X2, Y = self._generate_data(200, x1_min, x1_max, x2_min, x2_max)
        
        # Régression linéaire
        X = np.column_stack([X1, X2])
        model = LinearRegression()
        model.fit(X, Y)
        Y_pred = model.predict(X)
        
        # Métriques
        r2 = r2_score(Y, Y_pred)
        mse = mean_squared_error(Y, Y_pred)
        rmse = np.sqrt(mse)
        
        # Affichage formaté des métriques
        metrics_text = f"""R² = {r2:.4f}
MSE = {mse:.4f}
RMSE = {rmse:.4f}"""
        self.metrics_label.config(text=metrics_text)
        
        # Coefficients
        b0 = model.intercept_
        b1, b2 = model.coef_
        
        coeff_text = f"""β₀ = {b0:.4f}
β₁ = {b1:.4f}
β₂ = {b2:.4f}"""
        self.coeff_label.config(text=coeff_text)
        
        # Équation du modèle
        if b1 >= 0 and b2 >= 0:
            equation_text = f"Y = {b0:.4f} + {b1:.4f}·X₁ + {b2:.4f}·X₂"
        elif b1 >= 0 and b2 < 0:
            equation_text = f"Y = {b0:.4f} + {b1:.4f}·X₁ - {abs(b2):.4f}·X₂"
        elif b1 < 0 and b2 >= 0:
            equation_text = f"Y = {b0:.4f} - {abs(b1):.4f}·X₁ + {b2:.4f}·X₂"
        else:
            equation_text = f"Y = {b0:.4f} - {abs(b1):.4f}·X₁ - {abs(b2):.4f}·X₂"
        
        self.equation_label.config(text=equation_text)
        
        # Mettre à jour l'affichage
        self.update_idletasks()
        
        # Dessiner le graphique
        self._draw_chart(X1, X2, Y, Y_pred, mode)

    def _draw_chart(self, X1, X2, Y, Y_pred, mode):
        # Nettoyer ancien graphique
        for w in self.chart_frame.winfo_children():
            w.destroy()

        fig = plt.Figure(figsize=(9, 6), facecolor=COLORS["bg_card"])
        pastel = self.accent

        if mode == "2D":
            axes = [fig.add_subplot(1, 2, i+1) for i in range(2)]

            # Plot 1 : Y réel vs Y prédit
            ax = axes[0]
            ax.set_facecolor("#FAFBFC")
            ax.scatter(Y, Y_pred, alpha=0.55, color=pastel, edgecolors="#888", linewidths=0.4, s=25)
            mn, mx = min(Y.min(), Y_pred.min()), max(Y.max(), Y_pred.max())
            ax.plot([mn, mx], [mn, mx], "--", color="#B0B0B0", linewidth=1.2)
            ax.set_xlabel("Y réel", fontsize=9, color="#555")
            ax.set_ylabel("Y prédit", fontsize=9, color="#555")
            ax.set_title("Y réel vs Y prédit", fontsize=10, fontweight="bold", color="#333")
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

            # Plot 2 : résidus
            ax2 = axes[1]
            ax2.set_facecolor("#FAFBFC")
            residuals = Y - Y_pred
            ax2.scatter(Y_pred, residuals, alpha=0.55, color=TAB_COLORS["timeseries"],
                        edgecolors="#888", linewidths=0.4, s=25)
            ax2.axhline(0, color="#B0B0B0", linewidth=1, linestyle="--")
            ax2.set_xlabel("Y prédit", fontsize=9, color="#555")
            ax2.set_ylabel("Résidus", fontsize=9, color="#555")
            ax2.set_title("Analyse des Résidus", fontsize=10, fontweight="bold", color="#333")
            ax2.tick_params(labelsize=7)
            ax2.spines[["top", "right"]].set_visible(False)

        else:  # Mode 3D
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("#FAFBFC")
            
            # Surface de prédiction
            x1_surf = np.linspace(X1.min(), X1.max(), 30)
            x2_surf = np.linspace(X2.min(), X2.max(), 30)
            X1_surf, X2_surf = np.meshgrid(x1_surf, x2_surf)
            X_surf = np.column_stack([X1_surf.ravel(), X2_surf.ravel()])
            
            model = LinearRegression()
            model.fit(np.column_stack([X1, X2]), Y)
            Y_surf = model.predict(X_surf).reshape(X1_surf.shape)
            
            # Points de données réels
            ax.scatter(X1, X2, Y, alpha=0.6, color=pastel, 
                      edgecolors="#555", linewidths=0.3, s=30, label="Données réelles")
            
            # Surface de prédiction
            ax.plot_surface(X1_surf, X2_surf, Y_surf, alpha=0.5, color=TAB_COLORS["neuralnet"],
                           edgecolor='none', label="Surface de prédiction")
            
            ax.set_xlabel("X₁", fontsize=9, color="#555")
            ax.set_ylabel("X₂", fontsize=9, color="#555")
            ax.set_zlabel("Y", fontsize=9, color="#555")
            ax.set_title("Régression Linéaire Multiple 3D\n(X₁, X₂ → Y)", 
                        fontsize=11, fontweight="bold", color="#333")
            ax.legend(fontsize=8, loc='upper left')
            ax.tick_params(labelsize=7)
            ax.view_init(elev=25, azim=-60)

        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)