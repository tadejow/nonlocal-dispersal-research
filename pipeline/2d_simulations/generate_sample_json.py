import os
import json

# ==============================================================================
# DYNAMIC PATH SETUP
# ==============================================================================
# Zakładamy, że ten skrypt znajduje się w pipeline/2d_simulations/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
configs_dir = os.path.join(project_root, "configs")

# Upewnij się, że folder configs istnieje
os.makedirs(configs_dir, exist_ok=True)

# ==============================================================================
# DEFINE CONFIGURATIONS
# ==============================================================================

# Konfiguracja 1: Standardowe parametry
config_1 = {
    "A_max": 1.8,
    "A_min": 0.1,
    "B": 0.45,
    "d_u": 1.0,
    "d_v": 50.0,
    "L": 40.0,
    "N": 40,           # Rozmiar siatki
    "K_steps": 50,     # Liczba punktów na gałęzi
    "ht": 0.01,
    "tol": 1e-2,
    "max_iter": 50000
}

# ==============================================================================
# SAVE TO JSON
# ==============================================================================

file_1_path = os.path.join(configs_dir, "bifurcation_config_dv50.json")
with open(file_1_path, "w") as f:
    json.dump(config_1, f, indent=4)
print(f"Saved config to: {file_1_path}")

print("You can now run 'get_bifurcation_diagram.py'.")
