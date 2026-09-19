"""
=============================================================
  tabs/tab_cross_validation.py
  Onglet VI – Validation Croisée (Multi-Modèles)
  Affichage 4 quadrants : Accuracy, F1, Temps, Folds
  + Mode 2D/3D
=============================================================
"""

import tkinter as tk
import numpy as np
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa

from theme import COLORS, FONTS, TAB_COLORS
from utils.widgets import (Card, SectionTitle, ActionButton, Separator, 
                           DisplayToggle, FieldLabel, ScrollableSidebar)
from utils.data_generator import generate_dataset_regression


class CrossValidationTab(tk.Frame):
    def __init__(self, parent, go_to_page2=None):  # ← Ajout du callback
        """
        go_to_page2: callback pour revenir à la page 2 (grille des modules)
        """
        super().__init__(parent, bg=COLORS["bg_main"])
        self.accent = TAB_COLORS["crossval"]
        self.go_to_page2 = go_to_page2  # ← Stocker le callback
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chart_area()

    def _build_sidebar(self):
        # Utilisation du ScrollableSidebar
        self.sidebar_comp = ScrollableSidebar(self, width=280, bg=COLORS["bg_card"])
        self.sidebar_comp.grid(row=0, column=0, sticky="ns", padx=(16, 8), pady=16)
        
        sb = self.sidebar_comp.scrollable_frame
        p = 14

        # ── Bouton BACK vers Page 2 (AJOUTÉ) ─────────────────────
        back_btn = tk.Button(
            sb,
            text="◀ BACK to Modules",
            font=FONTS["body"],
            fg=COLORS["text_white"],
            bg=COLORS["accent_cv"],
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

        # Nombre de folds (seul paramètre restant)
        FieldLabel(sb, "Number of folds (K)").pack(anchor="w", padx=p, pady=(8, 2))
        self.folds_var = tk.IntVar(value=5)
        tk.Spinbox(sb, from_=2, to=10, textvariable=self.folds_var, width=6,
                   font=FONTS["body_small"], relief="flat", bg=COLORS["bg_main"],
                   highlightbackground=COLORS["border"], highlightthickness=1
                   ).pack(anchor="w", padx=p, pady=2)

        Separator(sb).pack(fill="x", padx=p, pady=12)

        # Mode d'affichage 2D/3D
        FieldLabel(sb, "Display mode").pack(anchor="w", padx=p, pady=(8, 2))
        self.display_toggle = DisplayToggle(sb)
        self.display_toggle.pack(anchor="w", padx=p, pady=4)

        Separator(sb).pack(fill="x", padx=p, pady=12)

        # Bouton d'exécution
        ActionButton(sb, "▶  Run ", self._run, accent=self.accent
                     ).pack(fill="x", padx=p, pady=(4, 16))

        # --- SECTION RÉSULTATS DÉTAILLÉS ---
        SectionTitle(sb, "📋  Detailed Results", accent_color=self.accent
                     ).pack(anchor="w", padx=p, pady=(8, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)
        
        self.scores_label = tk.Label(sb, text="—", font=FONTS["mono"],
                                      fg=COLORS["text_secondary"], bg=COLORS["bg_card"],
                                      justify="left", anchor="w")
        self.scores_label.pack(fill="x", padx=p, pady=(0, 16))

    def _go_back_to_page2(self):
        """Retourne à la Page 2 (grille des modules)"""
        if self.go_to_page2:
            self.go_to_page2()

    def _build_chart_area(self):
        self.chart_frame = Card(self)
        self.chart_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        tk.Label(self.chart_frame,
                 text="🧪  Click Run to compare models (Random Forest, MLP, SVM, Decision Tree)",
                 font=FONTS["subtitle"], fg=COLORS["text_secondary"],
                 bg=COLORS["bg_card"]).pack(expand=True)

    def _generate_data(self, n_samples=300):
        """Génère des données synthétiques pour la classification"""
        np.random.seed(42)
        # Génération de 3 features aléatoires
        X1 = np.random.uniform(-10, 10, n_samples)
        X2 = np.random.uniform(-5, 15, n_samples)
        X3 = np.random.uniform(0, 20, n_samples)
        
        X = np.column_stack([X1, X2, X3])
        
        # Classification binaire basée sur une combinaison linéaire avec bruit
        Y = (0.3 * X1 + 0.5 * X2 - 0.2 * X3 + np.random.randn(n_samples) * 2 > 0).astype(int)
        
        return X, Y

    def _run(self):
        folds = self.folds_var.get()
        mode = self.display_toggle.get()  # "2D" ou "3D"

        # Génération des données (dataset fixe pour la reproductibilité)
        X, Y = self._generate_data(300)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "Réseau Neurones": MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Arbre Décision": DecisionTreeClassifier(random_state=42)
        }

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        f1_scorer = make_scorer(f1_score, average='weighted')
        
        results = {}
        sidebar_text = ""

        for name, model in models.items():
            start = time.time()
            acc_scores = cross_val_score(model, X, Y, cv=cv, scoring="accuracy")
            f1_scores = cross_val_score(model, X, Y, cv=cv, scoring=f1_scorer)
            duration = (time.time() - start) / folds
            
            results[name] = {
                "acc": acc_scores,
                "f1": f1_scores,
                "time": duration
            }
            
            sidebar_text += f"{name.upper()}:\n"
            sidebar_text += f" • Acc: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}\n"
            sidebar_text += f" • F1:  {f1_scores.mean():.3f} ± {f1_scores.std():.3f}\n"
            sidebar_text += f" • Temps: {duration:.3f}s\n\n"

        self.scores_label.config(text=sidebar_text)
        self._draw_chart(results, folds, mode)

    def _draw_chart(self, results, folds, mode="2D"):
        # Nettoyer l'ancien graphique
        for w in self.chart_frame.winfo_children():
            w.destroy()

        if mode == "2D":
            fig = plt.Figure(figsize=(10, 7), facecolor=COLORS["bg_card"])
            names = list(results.keys())
            colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F1C40F"]  # Bleu, Rouge, Vert, Jaune
            
            # 1. Comparaison des exactitudes
            ax1 = fig.add_subplot(2, 2, 1)
            means_acc = [results[n]["acc"].mean() for n in names]
            stds_acc = [results[n]["acc"].std() for n in names]
            bars1 = ax1.bar(names, means_acc, yerr=stds_acc, color=colors, 
                            capsize=4, alpha=0.8, edgecolor='white', linewidth=1)
            ax1.set_title("Comparaison des exactitudes", fontsize=11, fontweight="bold", color='#333')
            ax1.set_ylabel("Exactitude moyenne", fontsize=9, color='#555')
            ax1.set_ylim(0, 1.1)
            for bar, val in zip(bars1, means_acc):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            # 2. Comparaison des scores F1
            ax2 = fig.add_subplot(2, 2, 2)
            means_f1 = [results[n]["f1"].mean() for n in names]
            stds_f1 = [results[n]["f1"].std() for n in names]
            bars2 = ax2.bar(names, means_f1, yerr=stds_f1, color=colors, 
                            capsize=4, alpha=0.8, edgecolor='white', linewidth=1)
            ax2.set_title("Comparaison des scores F1", fontsize=11, fontweight="bold", color='#333')
            ax2.set_ylabel("Score F1 moyen", fontsize=9, color='#555')
            ax2.set_ylim(0, 1.1)
            for bar, val in zip(bars2, means_f1):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            # 3. Temps d'entraînement
            ax3 = fig.add_subplot(2, 2, 3)
            times = [results[n]["time"] for n in names]
            bars3 = ax3.bar(names, times, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax3.set_title("Temps d'entraînement moyen", fontsize=11, fontweight="bold", color='#333')
            ax3.set_ylabel("Temps (secondes)", fontsize=9, color='#555')
            for bar, val in zip(bars3, times):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=8)

            # 4. Évolution par fold
            ax4 = fig.add_subplot(2, 2, 4)
            fold_x = np.arange(1, folds + 1)
            for i, name in enumerate(names):
                ax4.plot(fold_x, results[name]["acc"], marker='o', label=name, 
                        color=colors[i], linewidth=2, markersize=6)
            ax4.set_title("Évolution des performances par fold", fontsize=11, fontweight="bold", color='#333')
            ax4.set_xlabel("Fold", fontsize=9, color='#555')
            ax4.set_ylabel("Exactitude", fontsize=9, color='#555')
            ax4.legend(fontsize=7, loc='lower right')
            ax4.set_ylim(0, 1.1)
            ax4.grid(True, alpha=0.3)

            # Style commun pour tous les axes 2D
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis='x', rotation=15, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                ax.set_facecolor('#FAFBFC')

        else:  # Mode 3D
            fig = plt.Figure(figsize=(10, 8), facecolor=COLORS["bg_card"])
            names = list(results.keys())
            colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F1C40F"]
            
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor('#FAFBFC')
            
            # Préparer les données pour le graphique 3D
            x_pos = np.arange(len(names))
            y_pos = np.zeros(len(names))
            
            # Barres pour l'exactitude (hauteur = accuracy)
            for i, name in enumerate(names):
                mean_acc = results[name]["acc"].mean()
                std_acc = results[name]["acc"].std()
                ax.bar3d(x_pos[i], y_pos[i], 0, 0.6, 0.6, mean_acc,
                        color=colors[i], alpha=0.8, edgecolor='black')
            
            ax.set_xlabel("Modèles", fontsize=9, color='#555')
            ax.set_ylabel("", fontsize=9, color='#555')
            ax.set_zlabel("Exactitude moyenne", fontsize=9, color='#555')
            ax.set_title("Comparaison 3D des modèles\n(Validation Croisée)", 
                        fontsize=12, fontweight="bold", color='#333')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([n[:12] for n in names], rotation=15, fontsize=8)
            ax.set_zlim(0, 1.1)
            ax.view_init(elev=25, azim=-60)

        fig.tight_layout(pad=3.0)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)