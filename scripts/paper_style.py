"""Shared matplotlib style for paper figures: Times font, tight layout, PDF output."""
import matplotlib
import matplotlib.pyplot as plt

# Figures are created larger than display size and scaled by LaTeX.
# Font sizes are set large enough to remain readable after scaling.
# A figure created at 6.5" wide displayed at 3.25" scales fonts by 0.5x,
# so 16pt in the figure becomes ~8pt in the paper.
COLUMN_WIDTH = 3.25
TEXT_WIDTH = 6.75

def apply():
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

def savefig(fig, path):
    """Save as PDF with tight bounding box."""
    pdf_path = str(path).replace('.png', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    print(f"Saved: {pdf_path}")
    return pdf_path
