
# ── Palette de couleurs ──────────────────────────────────────
COLORS = {
    # Fond principal
    "bg_main":        "#F8F9FC",
    "bg_sidebar":     "#FFFFFF",
    "bg_card":        "#FFFFFF",
    "bg_header":      "#FFFFFF",

    # Accents pastel par onglet
    "accent_reg":     "#A8C5DA",   # bleu pastel  – Régression
    "accent_clust":   "#B8D4B8",   # vert pastel  – Clustering
    "accent_rf":      "#D4B8D4",   # violet pastel – Random Forest
    "accent_ts":      "#F4D4A8",   # pêche pastel  – Séries temp.
    "accent_nn":      "#D4A8A8",   # rose pastel   – Réseaux neurones
    "accent_cv":      "#A8D4D4",   # cyan pastel   – Validation croisée

    # Textes
    "text_primary":   "#2C3E50",
    "text_secondary": "#7F8C8D",
    "text_accent":    "#34495E",
    "text_white":     "#FFFFFF",

    # UI générale
    "border":         "#E8ECF0",
    "border_light":   "#F0F4F8",
    "shadow":         "#E0E4EA",
    "btn_hover":      "#EBF0F5",
    "success":        "#A8D4B8",
    "error":          "#D4A8A8",

    # Sidebar active
    "sidebar_active": "#EBF4FF",
    "sidebar_hover":  "#F5F8FC",
}

# ── Polices ──────────────────────────────────────────────────
FONTS = {
    "title":        ("Inter", 12, "bold"),
    "subtitle":     ("Georgia", 14, "bold"),
    "heading":      ("Helvetica", 12, "bold"),
    "body":         ("Helvetica", 10),
    "body_small":   ("Helvetica", 9),
    "label":        ("Helvetica", 10, "bold"),
    "mono":         ("Courier", 9),
    "sidebar_item": ("Helvetica", 11),
    "sidebar_title":("Georgia", 13, "bold"),
    "metric":       ("Georgia", 16, "bold"),
    "metric_label": ("Helvetica", 8),
}

# ── Dimensions ───────────────────────────────────────────────
LAYOUT = {
    "window_width":  1280,
    "window_height": 800,
    "sidebar_width": 220,
    "header_height": 70,
    "padding":       20,
    "card_radius":   8,
    "btn_padding":   (10, 20),
}

# ── Icônes texte pour chaque onglet ─────────────────────────
TAB_ICONS = {
    "regression":  "📈",
    "clustering":  "📊",
    "randomforest":"🌳",
    "timeseries":  "⏱",
    "neuralnet":   "🧠",
    "crossval":    "🔁",
}

TAB_LABELS = {
    "regression":   "Linear Regression",
    "clustering":   "K-Means Clustering",
    "randomforest": "Random Forest",
    "timeseries":   "Time Series",
    "neuralnet":    "Neural Networks",
    "crossval":     "Cross-Validation",
}

TAB_COLORS = {
    "regression":   COLORS["accent_reg"],
    "clustering":   COLORS["accent_clust"],
    "randomforest": COLORS["accent_rf"],
    "timeseries":   COLORS["accent_ts"],
    "neuralnet":    COLORS["accent_nn"],
    "crossval":     COLORS["accent_cv"],
}