# tabs/tab_welcome.py
import tkinter as tk
from theme import COLORS, FONTS, TAB_COLORS, TAB_ICONS, TAB_LABELS
from PIL import Image, ImageTk
import os
import sys


class WelcomeTab(tk.Frame):
    def __init__(self, parent, on_select, show_params=None):
        super().__init__(parent, bg=COLORS["bg_main"])
        self.on_select = on_select
        self.show_params = show_params
        self.current_state = "welcome"
        
        # Frame principal qui contiendra tout
        self.main_frame = tk.Frame(self, bg=COLORS["bg_main"])
        self.main_frame.pack(fill="both", expand=True)
        
        self._build_welcome_page()

    def _build_welcome_page(self):
        """Construit la page d'accueil (Page 1)"""
        # Nettoyer
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        self.current_state = "welcome"
        
        # Conteneur central
        center_frame = tk.Frame(self.main_frame, bg=COLORS["bg_main"])
        center_frame.pack(expand=True)
        
        # Carte principale
        card_frame = tk.Frame(
            center_frame,
            bg=COLORS["bg_card"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            bd=0,
            width=600,
            height=500
        )
        card_frame.pack(pady=50, padx=50)
        card_frame.pack_propagate(False)
        
        # Logo
        self._load_logo(card_frame)
        
        # Titre
        tk.Label(
            card_frame,
            text="Smart Application",
            font=("Inter", 28, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_card"]
        ).pack(pady=(10, 5))
        
        tk.Label(
            card_frame,
            text="Artificial Intelligence & Machine Learning",
            font=FONTS["subtitle"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_card"]
        ).pack(pady=(0, 20))
        
        # Séparateur
        separator = tk.Frame(card_frame, bg=COLORS["border"], height=1)
        separator.pack(fill="x", padx=40, pady=10)
        
        # Informations
        info_frame = tk.Frame(card_frame, bg=COLORS["bg_card"])
        info_frame.pack(pady=15)
        
        tk.Label(
            info_frame,
            text="📚 Student : Asala Hejji",
            font=FONTS["body"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_card"]
        ).pack()
        
        tk.Label(
            info_frame,
            text="👨‍🏫 Supervised by : EL MKHALET MOUNA",
            font=FONTS["body"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_card"]
        ).pack(pady=(5, 0))
        
        # Bouton START
        start_btn = tk.Button(
            card_frame,
            text="▶  START",
            font=FONTS["title"],
            fg=COLORS["text_white"],
            bg=COLORS["accent_reg"],
            activebackground=COLORS["accent_clust"],
            bd=0,
            padx=40,
            pady=12,
            cursor="hand2",
            command=self._show_page2
        )
        start_btn.pack(pady=(20, 30))
        
        # Effet hover
        def on_enter(e):
            start_btn.config(bg=COLORS["accent_clust"])
        def on_leave(e):
            start_btn.config(bg=COLORS["accent_reg"])
        start_btn.bind("<Enter>", on_enter)
        start_btn.bind("<Leave>", on_leave)

    def _show_page2(self):
        """Affiche la Page 2 : Grille des modules"""
        # Nettoyer complètement
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        self.current_state = "page2"
        
        # Conteneur principal
        main_container = tk.Frame(self.main_frame, bg=COLORS["bg_main"])
        main_container.pack(fill="both", expand=True)
        
        # En-tête avec bouton BACK
        header = tk.Frame(main_container, bg=COLORS["bg_header"], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        back_btn = tk.Button(
            header,
            text="◀  BACK",
            font=FONTS["body"],
            fg=COLORS["text_white"],
            bg=COLORS["accent_reg"],
            activebackground=COLORS["accent_clust"],
            bd=0,
            padx=15,
            pady=5,
            cursor="hand2",
            command=self._build_welcome_page  # Retour à Page 1
        )
        back_btn.pack(side="left", padx=20, pady=12)
        
        
        # Grille des modules
        grid_frame = tk.Frame(main_container, bg=COLORS["bg_main"])
        grid_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        # Configuration de la grille 2x3
        for i in range(2):
            grid_frame.rowconfigure(i, weight=1)
        for i in range(3):
            grid_frame.columnconfigure(i, weight=1)
        
        modules = [
            ("regression", "📈", "Linear Regression", "Linear modeling X → Y", 0, 0),
            ("clustering", "🔮", "K-Means Clustering", "Grouping into K clusters", 0, 1),
            ("randomforest", "🌲", "Random Forest", "Robust multi-tree prediction", 0, 2),
            ("timeseries", "📊", "Time Series", "Evolution & ARIMA forecasting", 1, 0),
            ("neuralnet", "🧠", "Neural Networks", "Complex non-linear relationships", 1, 1),
            ("crossval", "✅", "Cross-Validation", "Model comparison & selection", 1, 2),
        ]
        
        for key, icon, title, desc, row, col in modules:
            self._make_card(grid_frame, key, icon, title, desc).grid(
                row=row, column=col, padx=10, pady=10, sticky="nsew"
            )

    def _make_card(self, parent, key, icon, title, desc):
        """Crée une carte pour un module"""
        color = TAB_COLORS.get(key, COLORS["accent_reg"])
        
        card = tk.Frame(
            parent,
            bg=COLORS["bg_card"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            cursor="hand2",
            relief="flat"
        )
        
        # Bande colorée
        band = tk.Frame(card, bg=color, height=5)
        band.pack(fill="x")
        
        # Icône
        tk.Label(
            card,
            text=icon,
            font=("Segoe UI Emoji", 40),
            bg=COLORS["bg_card"]
        ).pack(pady=(20, 5))
        
        # Titre
        tk.Label(
            card,
            text=title,
            font=FONTS["heading"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_card"]
        ).pack()
        
        # Description
        tk.Label(
            card,
            text=desc,
            font=FONTS["body_small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_card"],
            wraplength=180
        ).pack(pady=(5, 15))
        
        # Bouton Open
        btn = tk.Button(
            card,
            text="Open →",
            font=FONTS["body_small"],
            fg=COLORS["text_white"],
            bg=color,
            activebackground=COLORS["btn_hover"],
            relief="flat",
            cursor="hand2",
            bd=0,
            padx=15,
            pady=5,
            command=lambda: self.on_select(key) if self.on_select else None
        )
        btn.pack(pady=(0, 20))
        
        # Effet hover
        def on_enter(e):
            card.config(highlightbackground=color, highlightthickness=2)
        def on_leave(e):
            card.config(highlightbackground=COLORS["border"], highlightthickness=1)
        
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        
        return card

    def _load_logo(self, parent):
        """Charge le logo"""
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            logo_path = os.path.join(base_path, "logo.png")
            
            if os.path.exists(logo_path):
                img = Image.open(logo_path)
                img = img.resize((80, 80), Image.Resampling.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(img)
                
                logo_label = tk.Label(parent, image=self.logo_image, bg=COLORS["bg_card"])
                logo_label.pack(pady=(20, 5))
        except Exception as e:
            print(f"Logo non chargé : {e}")