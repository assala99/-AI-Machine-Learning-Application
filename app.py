import tkinter as tk
from theme import (COLORS, FONTS, LAYOUT, TAB_ICONS, TAB_LABELS, TAB_COLORS)

from tabs.tab_welcome          import WelcomeTab
from tabs.tab_regression       import RegressionTab
from tabs.tab_clustering       import ClusteringTab
from tabs.tab_random_forest    import RandomForestTab
from tabs.tab_timeseries       import TimeSeriesTab
from tabs.tab_neural_network   import NeuralNetTab
from tabs.tab_cross_validation import CrossValidationTab

TAB_ORDER = ["regression", "clustering", "randomforest",
             "timeseries", "neuralnet", "crossval"]

TAB_CLASSES = {
    "regression":   RegressionTab,
    "clustering":   ClusteringTab,
    "randomforest": RandomForestTab,
    "timeseries":   TimeSeriesTab,
    "neuralnet":    NeuralNetTab,
    "crossval":     CrossValidationTab,
}


class MLApplication:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._active_tab_key = None
        self._current_frame  = None
        self._setup_window()
        self._build_layout()
        self._show_welcome()

    def _setup_window(self):
        w, h = 1280, 900
        self.root.title("Plateforme IA – Machine Learning ")
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(1100, 800)
    
    def _build_layout(self):
        # Une seule ligne pour le contenu
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        self._build_content_area()  # Pas de header, pas de tabbar

    def _build_content_area(self):
        self.content = tk.Frame(self.root, bg=COLORS["bg_main"])
        self.content.grid(row=0, column=0, sticky="nsew")
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

    def _show_welcome(self):
        self._clear_content()
        f = WelcomeTab(
            self.content, 
            on_select=self._activate_tab,
            show_params=None
        )
        f.grid(row=0, column=0, sticky="nsew")
        self._current_frame = f

    def _activate_tab(self, key):
        self._active_tab_key = key
        self._clear_content()
        
        tab_class = TAB_CLASSES[key]
        # Tous les onglets acceptent maintenant go_to_page2
        f = tab_class(self.content, go_to_page2=self._show_page2_from_model)
        
        f.grid(row=0, column=0, sticky="nsew")
        self._current_frame = f
        
    def _show_page2_from_model(self):
        """Retourne à la Page 2 depuis un modèle"""
        if self._current_frame:
            self._current_frame.destroy()
            self._current_frame = None
        
        f = WelcomeTab(
            self.content, 
            on_select=self._activate_tab,
            show_params=None
        )
        f.grid(row=0, column=0, sticky="nsew")
        self._current_frame = f
        f._show_page2()  # Afficher directement la grille

    def _clear_content(self):
        if self._current_frame:
            self._current_frame.destroy()
            self._current_frame = None