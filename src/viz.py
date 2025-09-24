import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_hist(df, save_path=None):
    plt.figure(figsize=(8,6))
    sns.histplot(df["finbert_neg"], kde=True, alpha=0.5, label="neg")
    sns.histplot(df["finbert_neu"], kde=True, alpha=0.5, label="neu")
    sns.histplot(df["finbert_pos"], kde=True, alpha=0.5, label="pos")
    plt.title("FinBERT sentiment probability distributions")
    plt.xlabel("Probability"); plt.legend()
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_label_counts(df, hue=None, save_path=None):
    plt.figure(figsize=(6,4))
    sns.countplot(x="finbert_label", hue=hue, data=df)
    plt.title("Label counts")
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_confidence_hist(df, save_path=None):
    plt.figure(figsize=(6,4))
    sns.histplot(df["finbert_confidence"], kde=True)
    plt.title("Prediction confidence distribution")
    plt.xlabel("Confidence")
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_pairwise(df, save_path=None):
    g = sns.pairplot(df[["finbert_neg","finbert_neu","finbert_pos","finbert_confidence"]], corner=True)
    g.fig.suptitle("Pairwise relationships", y=1.02)
    if save_path: g.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
