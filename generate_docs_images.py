import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.makedirs('docs', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

print("Generating docs images...")

# ── 1. SIGMOID FUNCTION ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
z = np.linspace(-10, 10, 400)
sigmoid = 1 / (1 + np.exp(-z))
ax.plot(z, sigmoid, color='#1a6fbd', linewidth=2.5, label='σ(z)')
ax.fill_between(z, sigmoid, 0.5, where=(z >= 0), alpha=0.12, color='#1a6fbd')
ax.fill_between(z, sigmoid, 0.5, where=(z <= 0), alpha=0.12, color='#e07b30')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(0,   color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(1.0, color='#aaa', linestyle=':', linewidth=0.8)
ax.axhline(0.0, color='#aaa', linestyle=':', linewidth=0.8)
ax.scatter([0], [0.5], color='#1a6fbd', s=60, zorder=5)
ax.annotate('σ(0) = 0.5', xy=(0, 0.5), xytext=(2.5, 0.32),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.2),
            fontsize=11, color='#333')
ax.set_xlabel('z', fontsize=13)
ax.set_ylabel('σ(z)', fontsize=13)
ax.set_title('Sigmoid Function   σ(z) = 1 / (1 + e⁻ᶻ)', fontsize=14, fontweight='bold', pad=14)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('docs/sigmoid_function.png', dpi=180, bbox_inches='tight')
plt.close()
print("1. sigmoid_function.png ✓")

# ── 2. DECISION BOUNDARY ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
np.random.seed(42)
n = 120
X0 = np.random.randn(n, 2) * 0.9 + [-1.8, -1.8]
X1 = np.random.randn(n, 2) * 0.9 + [1.8, 1.8]
ax.scatter(X0[:, 0], X0[:, 1], c='#1a6fbd', alpha=0.55, s=40, label='Class 0 (negative)', edgecolors='white', linewidth=0.5)
ax.scatter(X1[:, 0], X1[:, 1], c='#e07b30', alpha=0.55, s=40, label='Class 1 (positive)', edgecolors='white', linewidth=0.5)
x_line = np.linspace(-5, 5, 200)
y_line = -x_line
ax.plot(x_line, y_line, 'k-', linewidth=2, label='Decision boundary (θᵀx = 0)', zorder=5)
ax.fill_between(x_line, y_line, 6,  alpha=0.06, color='#e07b30')
ax.fill_between(x_line, y_line, -6, alpha=0.06, color='#1a6fbd')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('Logistic Regression — Decision Boundary', fontsize=14, fontweight='bold', pad=14)
ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('docs/decision_boundary.png', dpi=180, bbox_inches='tight')
plt.close()
print("2. decision_boundary.png ✓")

# ── 3. GRADIENT DESCENT 3D ───────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
theta0 = np.linspace(-3, 3, 80)
theta1 = np.linspace(-3, 3, 80)
T0, T1 = np.meshgrid(theta0, theta1)
Z = T0**2 + T1**2 + 0.3 * np.sin(4*T0) * np.cos(4*T1)
ax.plot_surface(T0, T1, Z, alpha=0.35, cmap='coolwarm', linewidth=0)
path_t0 = [2.5, 1.9, 1.4, 1.0, 0.65, 0.38, 0.18, 0.07, 0.01]
path_t1 = [2.5, 1.9, 1.4, 1.0, 0.65, 0.38, 0.18, 0.07, 0.01]
path_z  = [t**2 + t**2 + 0.3*np.sin(4*t)*np.cos(4*t) for t in path_t0]
ax.plot(path_t0, path_t1, path_z, 'yo-', linewidth=2, markersize=6, label='GD path', zorder=5)
ax.scatter([0.01], [0.01], [path_z[-1]], color='red', s=120, zorder=6, label='Minimum')
ax.set_title('Gradient Descent — Loss Surface', fontsize=13, fontweight='bold', pad=16)
ax.set_xlabel('θ₀', fontsize=11)
ax.set_ylabel('θ₁', fontsize=11)
ax.set_zlabel('Loss J(θ)', fontsize=11)
ax.legend(fontsize=10)
ax.view_init(elev=28, azim=-55)
plt.tight_layout()
plt.savefig('docs/gradient_descent_3d.png', dpi=180, bbox_inches='tight')
plt.close()
print("3. gradient_descent_3d.png ✓")

# ── 4. LOSS CURVE COMPARISON (GD vs Newton) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
gd_loss     = [0.693 * np.exp(-0.004 * i) + 0.12 for i in range(1000)]
newton_loss = [0.693 * np.exp(-0.6   * i) + 0.12 for i in range(12)]
axes[0].plot(range(1000), gd_loss,     color='#1a6fbd', linewidth=2, label='GD (1000 iters)')
axes[0].set_title('Gradient Descent', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[1].plot(range(12),   newton_loss, color='#e07b30', linewidth=2, marker='o', markersize=6, label='Newton (~11 iters)')
axes[1].set_title("Newton's Method", fontsize=13, fontweight='bold')
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('Loss')
axes[1].legend()
fig.suptitle('Convergence Comparison — GD vs Newton', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/loss_curve_comparison.png', dpi=180, bbox_inches='tight')
plt.close()
print("4. loss_curve_comparison.png ✓")

# ── 5. LOGISTIC VS LINEAR ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
np.random.seed(7)
x = np.linspace(-3, 3, 200)
true_prob = 1 / (1 + np.exp(-2 * x))
x_pts = np.random.uniform(-3, 3, 60)
y_pts = (1 / (1 + np.exp(-2 * x_pts)) > np.random.uniform(0, 1, 60)).astype(int)
# Linear regression on binary
m, b = np.polyfit(x_pts, y_pts, 1)
linear_pred = m * x + b
axes[0].scatter(x_pts, y_pts, color='#555', s=30, alpha=0.6, zorder=3)
axes[0].plot(x, linear_pred, color='red', linewidth=2, label='Linear fit')
axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1)
axes[0].fill_between(x, 0.5, linear_pred, where=(linear_pred > 1), alpha=0.2, color='red', label='Prediction > 1 (invalid)')
axes[0].fill_between(x, linear_pred, 0.5, where=(linear_pred < 0), alpha=0.2, color='red')
axes[0].set_ylim(-0.3, 1.3)
axes[0].set_title('Linear Regression on Binary Output\n(fails — predictions outside [0,1])', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Feature x')
axes[0].set_ylabel('Predicted value')
axes[0].legend(fontsize=9)
axes[1].scatter(x_pts, y_pts, color='#555', s=30, alpha=0.6, zorder=3)
axes[1].plot(x, true_prob, color='#1a6fbd', linewidth=2.5, label='Logistic (sigmoid)')
axes[1].axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Threshold 0.5')
axes[1].set_ylim(-0.1, 1.1)
axes[1].set_title('Logistic Regression on Binary Output\n(correct — bounded in [0,1])', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Feature x')
axes[1].set_ylabel('P(y=1 | x)')
axes[1].legend(fontsize=9)
fig.suptitle('Why Logistic Regression, Not Linear Regression', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/logistic_vs_linear.png', dpi=180, bbox_inches='tight')
plt.close()
print("5. logistic_vs_linear.png ✓")

# ── 6. PREPROCESSING PIPELINE ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')
steps = [
    ('Raw CSV', '#d4e6f7'),
    ('Drop\nuseless cols', '#cce5cc'),
    ('Handle\nnulls', '#cce5cc'),
    ('Encode\ntarget', '#cce5cc'),
    ('One-hot\nencode', '#cce5cc'),
    ('Train/Test\nsplit', '#fff3cc'),
    ('Remove\nmulticol.', '#fff3cc'),
    ('Fit\nscaler', '#fff3cc'),
    ('Scale\nfeatures', '#fff3cc'),
    ('Add\nbias col', '#f7d4d4'),
    ('Save\n.npy files', '#e8d5f5'),
]
box_w, box_h = 1.1, 1.2
start_x = 0.3
for i, (label, color) in enumerate(steps):
    x = start_x + i * 1.25
    rect = mpatches.FancyBboxPatch((x, 1.4), box_w, box_h,
                                    boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor='#888', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + box_w/2, 1.4 + box_h/2, label,
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#222')
    if i < len(steps) - 1:
        ax.annotate('', xy=(x + box_w + 0.14*1.25, 2.0), xytext=(x + box_w, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
legend_items = [
    mpatches.Patch(color='#d4e6f7', label='Input'),
    mpatches.Patch(color='#cce5cc', label='Clean & encode'),
    mpatches.Patch(color='#fff3cc', label='Split & scale'),
    mpatches.Patch(color='#f7d4d4', label='Bias'),
    mpatches.Patch(color='#e8d5f5', label='Output'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=5, fontsize=8.5,
          bbox_to_anchor=(0.5, -0.05), frameon=True)
ax.set_title('Preprocessing Pipeline — Applied to All 6 Datasets', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('docs/preprocessing_pipeline.png', dpi=180, bbox_inches='tight')
plt.close()
print("6. preprocessing_pipeline.png ✓")

# ── 7. DATASET OVERVIEW ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
datasets     = ['Breast\nCancer', 'Heart\nDisease', 'Diabetes', 'Stroke', 'Adult\nIncome', 'Credit\nFraud']
samples      = [569, 1025, 768, 5110, 32561, 284807]
features     = [21, 14, 9, 22, 109, 31]
imbalance    = [37, 46, 35, 5, 24, 0.17]
colors = ['#1a6fbd', '#e07b30', '#2eaa5e', '#9b59b6', '#e74c3c', '#f39c12']
bars0 = axes[0].bar(datasets, samples, color=colors, alpha=0.85, edgecolor='white')
axes[0].set_title('Sample Size', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Samples')
axes[0].set_yscale('log')
for bar, val in zip(bars0, samples):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                 f'{val:,}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
bars1 = axes[1].bar(datasets, features, color=colors, alpha=0.85, edgecolor='white')
axes[1].set_title('Feature Count (after preprocessing)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Features')
for bar, val in zip(bars1, features):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                 str(val), ha='center', va='bottom', fontsize=8.5, fontweight='bold')
bars2 = axes[2].bar(datasets, imbalance, color=colors, alpha=0.85, edgecolor='white')
axes[2].set_title('Minority Class % (imbalance)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Minority class %')
axes[2].axhline(20, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Imbalance threshold (20%)')
axes[2].legend(fontsize=8)
for bar, val in zip(bars2, imbalance):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                 f'{val}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
fig.suptitle('Dataset Overview — 6 Diverse Classification Challenges', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/dataset_overview.png', dpi=180, bbox_inches='tight')
plt.close()
print("7. dataset_overview.png ✓")

# ── 8. FINAL RESULTS HEATMAP ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
methods  = ['GD\n(no reg)', 'L1 Reg', 'L2 Reg', 'Threshold\nTuned', 'K-Fold CV\n(mean)', 'vs Sklearn']
datasets_h = ['Breast Cancer', 'Diabetes', 'Heart Disease', 'Credit Fraud', 'Stroke', 'Adult Income']
mcc_data = np.array([
    [0.9639, 0.9639, 0.9639, 0.9644, 0.9386, 0.9639],
    [0.5071, 0.5195, 0.5071, 0.5302, 0.4395, 0.5071],
    [0.7153, 0.7153, 0.7153, 0.7384, 0.7200, 0.7153],
    [0.0600, 0.0600, 0.0600, 0.0164, 0.6815, 0.0600],
    [0.0000, 0.0000, 0.0000, 0.2722, 0.0000, 0.1380],
    [0.5663, 0.5663, 0.5663, 0.5663, 0.5500, 0.5701],
])
sns.heatmap(mcc_data, annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=methods, yticklabels=datasets_h,
            vmin=0, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={'size': 10})
ax.set_title('MCC Scores Across All Methods and Datasets', fontsize=14, fontweight='bold', pad=14)
ax.set_xlabel('Method / Configuration', fontsize=11)
ax.set_ylabel('Dataset', fontsize=11)
plt.tight_layout()
plt.savefig('docs/final_results_heatmap.png', dpi=180, bbox_inches='tight')
plt.close()
print("8. final_results_heatmap.png ✓")

# ── 9. MCC VS ACCURACY ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
datasets_b   = ['Breast\nCancer', 'Diabetes', 'Heart\nDisease', 'Credit\nFraud', 'Stroke', 'Adult\nIncome']
accuracy     = [0.9825, 0.7727, 0.8537, 0.9978, 0.9511, 0.8500]
mcc_scores   = [0.9639, 0.5071, 0.7153, 0.0600, 0.0000, 0.5663]
x_pos = np.arange(len(datasets_b))
width = 0.35
bars_acc = ax.bar(x_pos - width/2, accuracy,   width, label='Accuracy', color='#1a6fbd', alpha=0.85, edgecolor='white')
bars_mcc = ax.bar(x_pos + width/2, mcc_scores, width, label='MCC',      color='#e07b30', alpha=0.85, edgecolor='white')
for bar in bars_acc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8.5, color='#1a6fbd', fontweight='bold')
for bar in bars_mcc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8.5, color='#e07b30', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(datasets_b, fontsize=10)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.12)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Accuracy vs MCC — Why Accuracy is Misleading on Imbalanced Data', fontsize=13, fontweight='bold', pad=14)
ax.legend(fontsize=11)
# Annotation for Credit Fraud
ax.annotate('Accuracy=0.998\nMCC=0.06\n← Model collapsed!',
            xy=(3 + width/2, 0.06), xytext=(3.6, 0.4),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
            fontsize=8.5, color='red', fontweight='bold')
ax.annotate('Accuracy=0.951\nMCC=0.000\n← Predicts majority\nclass only!',
            xy=(4 + width/2, 0.0), xytext=(4.1, 0.35),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
            fontsize=8.5, color='red', fontweight='bold')
plt.tight_layout()
plt.savefig('docs/mcc_vs_accuracy.png', dpi=180, bbox_inches='tight')
plt.close()
print("9. mcc_vs_accuracy.png ✓")

# ── 10. TRAINING PIPELINE ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 3.5)
ax.axis('off')
pipeline = [
    ('X_train\ny_train', '#d4e6f7'),
    ('Initialize\nθ = 0', '#cce5cc'),
    ('Compute\nσ(Xθ)', '#cce5cc'),
    ('Compute\nLoss J(θ)', '#fff3cc'),
    ('Compute\n∇J(θ)', '#fff3cc'),
    ('Update\nθ := θ - α∇J', '#f7d4d4'),
    ('Converged?\n|∇J| < tol', '#fde8d8'),
    ('Predict\nŷ = θᵀx', '#e8d5f5'),
    ('Evaluate\nF1, MCC', '#e8d5f5'),
]
box_w, box_h = 1.2, 1.1
start_x = 0.3
for i, (label, color) in enumerate(pipeline):
    x = start_x + i * 1.48
    rect = mpatches.FancyBboxPatch((x, 1.2), box_w, box_h,
                                    boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor='#888', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + box_w/2, 1.2 + box_h/2, label,
            ha='center', va='center', fontsize=8, fontweight='bold', color='#222')
    if i < len(pipeline) - 1:
        ax.annotate('', xy=(x + box_w + 0.18*1.48, 1.75), xytext=(x + box_w, 1.75),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.3))
# Loop back arrow for gradient descent
loop_x_start = start_x + 5 * 1.48 + box_w
loop_x_end   = start_x + 2 * 1.48
ax.annotate('', xy=(loop_x_end, 1.2), xytext=(loop_x_start, 1.2),
            arrowprops=dict(arrowstyle='->', color='#e07b30', lw=1.5,
                            connectionstyle='arc3,rad=0.35'))
ax.text((loop_x_start + loop_x_end)/2, 0.55, 'repeat until convergence',
        ha='center', va='center', fontsize=8, color='#e07b30', style='italic')
ax.set_title('Training Pipeline — Logistic Regression from Scratch', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('docs/training_pipeline.png', dpi=180, bbox_inches='tight')
plt.close()
print("10. training_pipeline.png ✓")

# ── 11. PROJECT ARCHITECTURE ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
def draw_box(ax, x, y, w, h, label, sublabel='', color='#d4e6f7', fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='#888', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#222')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha='center', va='center', fontsize=7.5, color='#555')
# Root
draw_box(ax, 4.5, 7.0, 3.0, 0.7, 'logistic-regression-from-scratch', color='#e8d5f5', fontsize=8)
# src
draw_box(ax, 0.3, 5.2, 2.5, 1.4, 'src/', 'core implementation', color='#cce5cc', fontsize=9)
ax.text(0.5, 5.6, '• preprocessing.py', fontsize=7.5, color='#333')
ax.text(0.5, 5.3, '• logistic_master.py', fontsize=7.5, color='#333')
ax.text(0.5, 5.0, '• utils.py', fontsize=7.5, color='#333')
# Theory
draw_box(ax, 3.2, 5.2, 2.5, 1.4, 'Theory/', '6 LaTeX notebooks', color='#fff3cc', fontsize=9)
ax.text(3.4, 5.6, '• Bernoulli MLE', fontsize=7.5, color='#333')
ax.text(3.4, 5.3, '• Hessian PSD', fontsize=7.5, color='#333')
ax.text(3.4, 5.0, '• Fisher Info...', fontsize=7.5, color='#333')
# Notebooks
draw_box(ax, 6.1, 5.2, 2.5, 1.4, 'Notebooks/', '10 experiment notebooks', color='#d4e6f7', fontsize=9)
ax.text(6.3, 5.6, '• 00 Engine validation', fontsize=7.5, color='#333')
ax.text(6.3, 5.3, '• 05 Regularization', fontsize=7.5, color='#333')
ax.text(6.3, 5.0, '• 10 Sklearn compare', fontsize=7.5, color='#333')
# Data
draw_box(ax, 9.0, 5.2, 2.5, 1.4, 'Data/', 'raw + processed', color='#fde8d8', fontsize=9)
ax.text(9.2, 5.6, '• 6 raw CSVs (155MB)', fontsize=7.5, color='#333')
ax.text(9.2, 5.3, '• 24 .npy arrays', fontsize=7.5, color='#333')
ax.text(9.2, 5.0, '• 6 datasets', fontsize=7.5, color='#333')
# Results
draw_box(ax, 3.5, 3.4, 5.0, 1.4, 'Results/', 'figures + metrics + models', color='#f7d4d4', fontsize=9)
ax.text(3.7, 3.8, '• 11 figures (PNG)', fontsize=7.5, color='#333')
ax.text(3.7, 3.5, '• final_metrics.json', fontsize=7.5, color='#333')
ax.text(6.5, 3.8, '• 6 theta .npy files', fontsize=7.5, color='#333')
ax.text(6.5, 3.5, '• Phase summaries', fontsize=7.5, color='#333')
# Phases label
draw_box(ax, 1.5, 1.5, 9.0, 1.5, '', color='#f5f5f5', fontsize=9)
phases = ['I\nSetup', 'II\nTheory', 'III\nPreproc', 'IV\nTraining', 'V\nReg', 'VI\nDiag', 'VII\nInfer', 'VIII\nFailure', 'IX\nAdv', 'X\nSklearn']
phase_colors = ['#cce5cc']*2 + ['#fff3cc']*2 + ['#d4e6f7']*2 + ['#fde8d8']*2 + ['#e8d5f5']*2
for j, (ph, pc) in enumerate(zip(phases, phase_colors)):
    px = 1.7 + j * 0.88
    rect = mpatches.FancyBboxPatch((px, 1.6), 0.78, 1.2,
                                    boxstyle="round,pad=0.06",
                                    facecolor=pc, edgecolor='#aaa', linewidth=0.6)
    ax.add_patch(rect)
    ax.text(px + 0.39, 2.2, ph, ha='center', va='center', fontsize=7, fontweight='bold', color='#222')
ax.text(6.0, 1.2, '10 phases — theory → preprocessing → training → inference → sklearn comparison',
        ha='center', va='center', fontsize=8, color='#555', style='italic')
# Arrows from root to main folders
for tx in [1.55, 4.45, 7.35, 10.25]:
    ax.annotate('', xy=(tx, 6.6), xytext=(6.0, 7.0),
                arrowprops=dict(arrowstyle='->', color='#888', lw=0.8))
ax.set_title('Project Architecture — Logistic Regression from Scratch', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('docs/project_architecture.png', dpi=180, bbox_inches='tight')
plt.close()
print("11. project_architecture.png ✓")

print("\nAll 11 images saved to docs/ folder.")
print("Files:")
for f in sorted(os.listdir('docs')):
    print(f"  {f}")