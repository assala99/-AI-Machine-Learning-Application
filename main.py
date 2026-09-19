"""
=============================================================
  Application IA - Machine Learning Desktop
  Dr. EL MKHALET MOUNA - EMSI
  main.py : Point d'entrée principal
=============================================================
"""
import sys
import os
import tkinter as tk
from app import MLApplication

def resource_path(relative_path):
    """ Récupère le chemin absolu vers la ressource, compatible PyInstaller """
    try:
        # PyInstaller crée un dossier temporaire _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

#  ces lignes pour aider Python à trouver vos modules
sys.path.append(resource_path("."))
sys.path.append(resource_path("tabs"))
sys.path.append(resource_path("utils"))

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApplication(root)
    root.mainloop()