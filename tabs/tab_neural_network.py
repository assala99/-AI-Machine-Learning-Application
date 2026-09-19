import tkinter as tk
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

from theme import COLORS, FONTS, TAB_COLORS
from utils.widgets import (Card, SectionTitle, MinMaxControl,
                            ActionButton, MetricBox, Separator, DisplayToggle, FieldLabel, ScrollableSidebar)

class NeuralNetTab(tk.Frame):
    def __init__(self, parent, go_to_page2=None):
        super().__init__(parent, bg=COLORS["bg_main"])
        self.accent = TAB_COLORS["neuralnet"]
        self.go_to_page2 = go_to_page2
        self.current_view = "3D"
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_chart_area()
        self._build_sidebar()

    def _build_sidebar(self):
        self.sidebar_comp = ScrollableSidebar(self, width=280, bg=COLORS["bg_card"])
        self.sidebar_comp.grid(row=0, column=0, sticky="ns", padx=(16, 8), pady=16)
        
        sb = self.sidebar_comp.scrollable_frame
        p = 14

        # BACK button
        back_btn = tk.Button(
            sb,
            text="◀ BACK to Modules",
            font=FONTS["body"],
            fg=COLORS["text_white"],
            bg=COLORS["accent_nn"],
            activebackground=COLORS["accent_clust"],
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._go_back_to_page2
        )
        back_btn.pack(anchor="w", padx=p, pady=(8, 8))

        SectionTitle(sb, "⚙️ Parameters", accent_color=self.accent).pack(anchor="w", padx=p, pady=(16, 4))
        Separator(sb).pack(fill="x", padx=p, pady=4)

        FieldLabel(sb, "Data Ranges (X1, X2, X3)").pack(anchor="w", padx=p, pady=(8, 2))
        self.x1 = MinMaxControl(sb, "X1", -10, 10)
        self.x1.pack(anchor="w", padx=p, pady=2)
        self.x2 = MinMaxControl(sb, "X2", -5, 15)
        self.x2.pack(anchor="w", padx=p, pady=2)
        self.x3 = MinMaxControl(sb, "X3", 0, 20)
        self.x3.pack(anchor="w", padx=p, pady=2)

        Separator(sb).pack(fill="x", padx=p, pady=8)

        FieldLabel(sb, "Hidden layers").pack(anchor="w", padx=p, pady=(0, 2))
        self.layers_var = tk.StringVar(value="64, 32")
        tk.Entry(sb, textvariable=self.layers_var, font=FONTS["body_small"],
                 relief="flat", bg=COLORS["bg_main"], highlightthickness=1).pack(fill="x", padx=p, pady=2)
        
        FieldLabel(sb, "Max Iterations").pack(anchor="w", padx=p, pady=(8, 2))
        self.iter_var = tk.IntVar(value=500)
        tk.Spinbox(sb, from_=100, to=5000, textvariable=self.iter_var, width=8,
                   font=FONTS["body_small"], relief="flat", bg=COLORS["bg_main"]).pack(anchor="w", padx=p, pady=2)

        Separator(sb).pack(fill="x", padx=p, pady=8)
        
        # 2D/3D Toggle
        self.display_toggle = DisplayToggle(sb, default="3D")
        self.display_toggle.pack(anchor="w", padx=p, pady=4)
        self.display_toggle.mode_var.trace_add('write', self._on_view_change)

        ActionButton(sb, "▶ Run", self._run, accent=self.accent).pack(fill="x", padx=p, pady=(4, 16))

        SectionTitle(sb, "📊 Performance", accent_color=self.accent).pack(anchor="w", padx=p, pady=(8, 4))
        self.m_loss = MetricBox(sb, "Loss finale", accent=self.accent)
        self.m_r2 = MetricBox(sb, "R² Score", accent=self.accent)
        self.m_mse = MetricBox(sb, "MSE", accent=self.accent)
        self.m_loss.pack(fill="x", padx=p, pady=3)
        self.m_r2.pack(fill="x", padx=p, pady=3)
        self.m_mse.pack(fill="x", padx=p, pady=3)

        # Stockage des résultats
        self.last_X_test = None
        self.last_Y_test = None
        self.last_Y_pred_test = None
        self.last_loss_curve = None
        self.last_model = None
        self.last_X_scaled = None

    def _on_view_change(self, *args):
        self.current_view = self.display_toggle.get()
        if self.last_X_test is not None:
            if self.current_view == "3D":
                self._draw_3d_chart()
            else:
                self._draw_2d_chart()

    def _go_back_to_page2(self):
        if self.go_to_page2:
            self.go_to_page2()

    def _build_chart_area(self):
        self.chart_frame = Card(self)
        self.chart_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        
        self.placeholder_label = tk.Label(self.chart_frame, 
                 text="🧠 Configure the architecture and click Run\n\n📊 Neural Network Regression with X1, X2, X3",
                 font=FONTS["subtitle"], fg=COLORS["text_secondary"], bg=COLORS["bg_card"],
                 justify="center")
        self.placeholder_label.pack(expand=True)

    def _generate_3d_data(self, n_samples=800):
        x1_min, x1_max = self.x1.get()
        x2_min, x2_max = self.x2.get()
        x3_min, x3_max = self.x3.get()
        
        X1 = np.random.uniform(x1_min, x1_max, n_samples)
        X2 = np.random.uniform(x2_min, x2_max, n_samples)
        X3 = np.random.uniform(x3_min, x3_max, n_samples)
        
        # Fonction cible
        Y = (np.sin(X1) * np.cos(X2) + 
             0.3 * X3**2 / 10 + 
             0.5 * X1 * X2 / 20 + 
             np.random.normal(0, 0.3, n_samples))
        
        Y = (Y - np.mean(Y)) / np.std(Y)
        
        return np.column_stack([X1, X2, X3]), Y

    def _run(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()

        try:
            layers = tuple(map(int, self.layers_var.get().replace(' ', '').split(',')))
            max_iter = self.iter_var.get()
            
            X, Y = self._generate_3d_data(n_samples=800)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = MLPRegressor(
                hidden_layer_sizes=layers,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
            
            n_train = int(0.8 * len(X))
            idx = np.random.permutation(len(X))
            X_train, X_test = X_scaled[idx[:n_train]], X_scaled[idx[n_train:]]
            Y_train, Y_test = Y[idx[:n_train]], Y[idx[n_train:]]
            
            model.fit(X_train, Y_train)
            Y_pred_test = model.predict(X_test)
            
            from sklearn.metrics import mean_squared_error, r2_score
            test_mse = mean_squared_error(Y_test, Y_pred_test)
            r2 = r2_score(Y_test, Y_pred_test)
            
            self.m_loss.set(f"{model.loss_:.4f}")
            self.m_r2.set(f"{r2:.3f}")
            self.m_mse.set(f"{test_mse:.4f}")
            
            # Stockage
            self.last_X_test = X_test
            self.last_Y_test = Y_test
            self.last_Y_pred_test = Y_pred_test
            self.last_loss_curve = model.loss_curve_
            self.last_model = model
            self.last_X_scaled = X_scaled
            
            if self.current_view == "3D":
                self._draw_3d_chart()
            else:
                self._draw_2d_chart()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_label = tk.Label(self.chart_frame,
                                  text=f"❌ Error: {str(e)}",
                                  font=FONTS["body"],
                                  fg="red",
                                  bg=COLORS["bg_card"])
            error_label.pack(expand=True)

    def _draw_2d_chart(self):
        """Vue 2D: Actual vs Predicted + Training Loss Curve"""
        fig = plt.Figure(figsize=(10, 6), facecolor=COLORS["bg_card"])
        
        # Graphique 1: Actual vs Predicted
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(self.last_Y_test, self.last_Y_pred_test, alpha=0.5, 
                   color=self.accent, s=15, edgecolors='white', linewidth=0.5)
        
        # Ligne diagonale y=x
        min_val = min(self.last_Y_test.min(), self.last_Y_pred_test.min())
        max_val = max(self.last_Y_test.max(), self.last_Y_pred_test.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect prediction')
        
        ax1.set_xlabel('Actual', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted', fontsize=11, fontweight='bold')
        ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Training Loss Curve
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(self.last_loss_curve, color=self.accent, linewidth=2.5)
        ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax2.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout(pad=3.0)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_3d_chart(self):
        """Vue 3D: Actual Y & Prediction Error avec X1, X2"""
        fig = plt.Figure(figsize=(10, 8), facecolor=COLORS["bg_card"])
        
        # Graphique 3D: Points réels
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        n_display = min(300, len(self.last_X_test))
        idx_sample = np.random.choice(len(self.last_X_test), n_display, replace=False)
        
        # Échantillon des données pour affichage
        X1_sample = self.last_X_test[idx_sample, 0]
        X2_sample = self.last_X_test[idx_sample, 1]
        Y_real_sample = self.last_Y_test[idx_sample]
        
        scatter1 = ax1.scatter(X1_sample, X2_sample, Y_real_sample, 
                              c=Y_real_sample, cmap='viridis', 
                              s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax1.set_xlabel('X1', fontsize=10, fontweight='bold', labelpad=8)
        ax1.set_ylabel('X2', fontsize=10, fontweight='bold', labelpad=8)
        ax1.set_zlabel('Y', fontsize=10, fontweight='bold', labelpad=8)
        ax1.set_title('Actual Y (3D View)', fontsize=12, fontweight='bold')
        fig.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=10)
        
        # Graphique 3D: Prédictions avec erreur
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        Y_pred_sample = self.last_Y_pred_test[idx_sample]
        errors = np.abs(Y_real_sample - Y_pred_sample)
        
        scatter2 = ax2.scatter(X1_sample, X2_sample, Y_pred_sample, 
                              c=errors, cmap='plasma', 
                              s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax2.set_xlabel('X1', fontsize=10, fontweight='bold', labelpad=8)
        ax2.set_ylabel('X2', fontsize=10, fontweight='bold', labelpad=8)
        ax2.set_zlabel('Y', fontsize=10, fontweight='bold', labelpad=8)
        ax2.set_title('Predicted Y (color = |error|)', fontsize=12, fontweight='bold')
        
        # Colorbar pour l'erreur
        cbar = fig.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=10)
        cbar.set_label('Absolute Error', fontsize=9, fontweight='bold')
        
        fig.tight_layout(pad=3.0)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)