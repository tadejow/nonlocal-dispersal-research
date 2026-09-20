import numpy as np
import matplotlib.pyplot as plt
import os

class SpectrumPlotter:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_beta_vs_L(self, L_vals, beta_vals, top_k=6, filename="beta_vs_L.png"):
        plt.figure(figsize=(10, 6))
        
        beta_array = np.array(beta_vals) 
        
        for k in range(min(top_k, beta_array.shape[1])):
            label_str = r"$\beta_{" + str(k) + r"}$"
            plt.plot(L_vals, beta_array[:, -(k+1)], marker='o', label=label_str)
            
        plt.title("Principal Eigenvalues vs Domain Size $")
        plt.xlabel("Domain Size $")
        plt.ylabel(r"Eigenvalue $\beta$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_eigenfunctions_gallery(self, eigenvalues, eigenfunctions_eval, nodes, filename="eigenfunctions_gallery.png"):
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        # Sort eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        
        for k in range(min(6, len(eigenvalues))):
            ax = axes[k]
            lam = eigenvalues[idx[k]]
            efunc = eigenfunctions_eval[:, idx[k]]
            
            # Normalize sign for consistent plotting
            if np.abs(np.min(efunc)) > np.max(efunc):
                efunc = -efunc
                
            ax.plot(nodes, efunc, lw=2)
            title_str = r"$\beta_{" + str(k) + r"} = " + f"{lam:.4f}$"
            ax.set_title(title_str)
            ax.grid(True)
            ax.set_xlabel(r"$")
            ax.set_ylabel(r"(x)$")
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
