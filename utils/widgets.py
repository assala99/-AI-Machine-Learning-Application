import tkinter as tk
from tkinter import ttk
from theme import COLORS, FONTS, LAYOUT


# ── Card blanche avec bordure douce ─────────────────────────
class Card(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=COLORS["bg_card"],
            relief="flat",
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            **kwargs
        )


# ── Titre de section ─────────────────────────────────────────
class SectionTitle(tk.Label):
    def __init__(self, parent, text, accent_color=None, **kwargs):
        color = accent_color or COLORS["text_primary"]
        super().__init__(
            parent,
            text=text,
            font=FONTS["subtitle"],
            fg=color,
            bg=parent["bg"] if "bg" in parent.keys() else COLORS["bg_card"],
            **kwargs
        )


# ── Label standard ───────────────────────────────────────────
class FieldLabel(tk.Label):
    def __init__(self, parent, text, **kwargs):
        super().__init__(
            parent,
            text=text,
            font=FONTS["label"],
            fg=COLORS["text_accent"],
            bg=parent["bg"] if "bg" in parent.keys() else COLORS["bg_card"],
            **kwargs
        )


# ── Champ min/max avec spinbox ───────────────────────────────
class MinMaxControl(tk.Frame):
    def __init__(self, parent, label, min_default, max_default,
                 min_range=(-1000, 1000), max_range=(-1000, 1000), **kwargs):
        bg = parent["bg"] if "bg" in parent.keys() else COLORS["bg_card"]
        super().__init__(parent, bg=bg, **kwargs)

        # On utilise StringVar pour éviter les erreurs de typage au démarrage
        self.min_var = tk.StringVar(value=str(min_default))
        self.max_var = tk.StringVar(value=str(max_default))

        # Label X1/X2 resserré (width=4)
        tk.Label(self, text=label, font=FONTS["label"],
                 fg=COLORS["text_primary"], bg=bg, width=4, anchor="w"
                 ).grid(row=0, column=0, padx=(0, 2))

        tk.Label(self, text="Min:", font=FONTS["body_small"],
                 fg=COLORS["text_secondary"], bg=bg
                 ).grid(row=0, column=1, padx=(0, 2))
        
        tk.Spinbox(self, from_=min_range[0], to=min_range[1],
                   textvariable=self.min_var, width=5, font=FONTS["body_small"],
                   relief="flat", bg=COLORS["bg_main"],
                   highlightbackground=COLORS["border"], highlightthickness=1
                   ).grid(row=0, column=2, padx=(0, 6))

        tk.Label(self, text="Max:", font=FONTS["body_small"],
                 fg=COLORS["text_secondary"], bg=bg
                 ).grid(row=0, column=3, padx=(0, 2))
        
        tk.Spinbox(self, from_=max_range[0], to=max_range[1],
                   textvariable=self.max_var, width=5, font=FONTS["body_small"],
                   relief="flat", bg=COLORS["bg_main"],
                   highlightbackground=COLORS["border"], highlightthickness=1
                   ).grid(row=0, column=4)

    def get(self):
        # On force la conversion en float ici pour être sûr
        try:
            return float(self.min_var.get()), float(self.max_var.get())
        except:
            return 0.0, 10.0 # Valeurs par défaut en cas d'erreur de saisie
# ── Bouton principal coloré ──────────────────────────────────
class ActionButton(tk.Button):
    def __init__(self, parent, text, command, accent=None, **kwargs):
        color = accent or COLORS["accent_reg"]
        super().__init__(
            parent,
            text=text,
            command=command,
            font=FONTS["heading"],
            fg=COLORS["text_primary"],
            bg=color,
            activebackground=COLORS["btn_hover"],
            activeforeground=COLORS["text_primary"],
            relief="flat",
            cursor="hand2",
            padx=LAYOUT["btn_padding"][1],
            pady=LAYOUT["btn_padding"][0],
            bd=0,
            **kwargs
        )
        self.bind("<Enter>", lambda e: self.config(bg=COLORS["btn_hover"]))
        self.bind("<Leave>", lambda e: self.config(bg=color))


# ── Métrique affichée en grand ───────────────────────────────
class MetricBox(tk.Frame):
    def __init__(self, parent, label, value="—", accent=None, **kwargs):
        bg = COLORS["bg_card"]
        super().__init__(parent, bg=bg,
                         highlightbackground=COLORS["border"],
                         highlightthickness=1, **kwargs)
        self.value_var = tk.StringVar(value=str(value))
        color = accent or COLORS["accent_reg"]

        tk.Label(self, textvariable=self.value_var,
                 font=FONTS["metric"], fg=color, bg=bg
                 ).pack(pady=(10, 2))
        tk.Label(self, text=label, font=FONTS["metric_label"],
                 fg=COLORS["text_secondary"], bg=bg
                 ).pack(pady=(0, 10))

    def set(self, value):
        self.value_var.set(str(value))


# ── Séparateur horizontal ────────────────────────────────────
class Separator(tk.Frame):
    def __init__(self, parent, **kwargs):
        bg = parent["bg"] if "bg" in parent.keys() else COLORS["bg_card"]
        super().__init__(parent, height=1,
                         bg=COLORS["border"], **kwargs)


# ── Radio toggle 2D / 3D ─────────────────────────────────────
class DisplayToggle(tk.Frame):
    """Boutons radio pour choisir le mode d'affichage 2D ou 3D."""
    def __init__(self, parent, default="2D", **kwargs):
        bg = parent["bg"] if "bg" in parent.keys() else COLORS["bg_card"]
        super().__init__(parent, bg=bg, **kwargs)

        self.mode_var = tk.StringVar(value=default)

        tk.Label(self, text="Display :", font=FONTS["label"],
                 fg=COLORS["text_secondary"], bg=bg
                 ).pack(side="left", padx=(0, 8))

        for mode in ("2D", "3D"):
            tk.Radiobutton(
                self, text=mode, variable=self.mode_var, value=mode,
                font=FONTS["body"], fg=COLORS["text_primary"], bg=bg,
                activebackground=bg, selectcolor=COLORS["bg_main"],
                relief="flat", cursor="hand2"
            ).pack(side="left", padx=4)

    def get(self):
        return self.mode_var.get()
    
#scroller    
class ScrollableSidebar(tk.Frame):
    def __init__(self, parent, width=280, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Conteneur principal (la Card)
        self.container = tk.Frame(self, width=width, bg=COLORS["bg_card"],
                                  highlightbackground=COLORS["border"], 
                                  highlightthickness=1)
        self.container.pack(fill="both", expand=True)
        self.container.pack_propagate(False)

        # Canvas pour le défilement
        self.canvas = tk.Canvas(self.container, bg=COLORS["bg_card"], 
                                highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.container, orient="vertical", 
                                      command=self.canvas.yview)
        
        # Le Frame qui contiendra réellement vos widgets (Paramètres, Métriques)
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLORS["bg_card"])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Fenêtre dans le canvas
        self.canvas_window = self.canvas.create_window((0, 0), 
                                                        window=self.scrollable_frame, 
                                                        anchor="nw", 
                                                        width=width-20) # -20 pour la scrollbar

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Placement
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Permettre le défilement avec la molette de la souris
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")