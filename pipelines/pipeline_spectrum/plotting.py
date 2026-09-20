import numpy as np
import matplotlib.pyplot as plt
import os

class SpectrumPlotter:
    def __init__(self, output_dir="output"):
        # Strip potential hidden unicode characters or whitespace
        self.output_dir = os.path.abspath(output_dir.strip('\u202a\u202b\u202c\u202d\u202e\t\r\n '))
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_beta_vs_L(self, L_vals, beta_vals, ess_bounds=None, top_k=100, filename="beta_vs_L.png"):
        plt.figure(figsize=(12, 7))
        
        beta_array = np.array(beta_vals) 
        k_actual = min(top_k, beta_array.shape[1])
        
        # Plot eigenvalues as a dense scatter plot (acts like a heatmap/spectrum diagram)
        for i, L in enumerate(L_vals):
            evals = beta_array[i, -k_actual:]
            # Using 'o' marker with larger size to be clearly visible
            plt.scatter(evals, [L]*len(evals), c='red', s=25, marker='o', alpha=0.6, edgecolors='none')
            
        if ess_bounds is not None:
            plt.plot(ess_bounds, L_vals, color='black', linewidth=2, linestyle='--', label=r'Ess. Spectrum Bound ($\sup \sigma_{ess}$)')
            plt.fill_betweenx(L_vals, ess_bounds, -2.0, color='gray', alpha=0.2, label='Essential Spectrum')
            plt.legend()
            
        plt.title(f"Non-local dispersal spectrum vs domain size")
        plt.xlabel(r"Eigenvalue $\beta$")
        plt.ylabel("Domain Size $L$")
        
        # Limit beta axis to -1.0 on the left
        plt.xlim([-1.0, 0.05])
        
        # Optionally, mark the boundaries
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save plot
        out_path = os.path.join(os.path.abspath(self.output_dir.strip()), filename.strip())
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
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
