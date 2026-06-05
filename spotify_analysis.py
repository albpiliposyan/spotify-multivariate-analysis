"""Generate multivariate analysis tables and figures for the Spotify dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "spotify_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

RANDOM_STATE = 42
PAIRPLOT_SAMPLE_SIZE = 3_000
QQ_SAMPLE_SIZE = 5_000
PROJECTION_SAMPLE_SIZE = 10_000

SELECTED_VARIABLES = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

PREDICTORS = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

PAIRPLOT_VARIABLES = [
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "valence",
]

GAUSSIAN_DIAGNOSTIC_VARIABLES = [
    "popularity",
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "tempo",
]


def prepare_output_directories() -> None:
    """Create folders for generated figures and tables."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean_data() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the CSV and keep one complete row per Spotify track."""
    raw = pd.read_csv(DATA_PATH)
    raw = raw.drop(columns=["Unnamed: 0"], errors="ignore")

    cleaning_summary = {
        "original_rows": len(raw),
        "original_columns": raw.shape[1],
        "duplicate_track_ids": int(raw.duplicated(subset="track_id").sum()),
        "missing_selected_values": int(raw[SELECTED_VARIABLES].isna().sum().sum()),
        "genre_count": int(raw["track_genre"].nunique()),
    }

    data = (
        raw.drop_duplicates(subset="track_id")
        .dropna(subset=SELECTED_VARIABLES)
        .copy()
    )
    cleaning_summary["nonpositive_duration_rows"] = int(
        (data["duration_ms"] <= 0).sum()
    )
    data = data.loc[data["duration_ms"] > 0].copy()
    cleaning_summary["analysis_rows"] = len(data)

    return data, cleaning_summary


def save_descriptive_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save descriptive statistics, skewness, and the correlation matrix."""
    descriptive = (
        data[SELECTED_VARIABLES]
        .describe()
        .T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    )
    descriptive.index.name = "variable"
    descriptive.to_csv(TABLE_DIR / "descriptive_statistics.csv")

    distribution_summary = pd.DataFrame(
        {
            "skewness": data[SELECTED_VARIABLES].skew(),
            "excess_kurtosis": data[SELECTED_VARIABLES].kurtosis(),
        }
    )
    distribution_summary.index.name = "variable"
    distribution_summary.to_csv(TABLE_DIR / "distribution_shape.csv")

    correlation = data[SELECTED_VARIABLES].corr()
    correlation.to_csv(TABLE_DIR / "correlation_matrix.csv")

    return descriptive, correlation


def create_pairplot(data: pd.DataFrame) -> None:
    """Create representative pairwise scatter plots using a random sample."""
    sample = data[PAIRPLOT_VARIABLES].sample(
        n=min(PAIRPLOT_SAMPLE_SIZE, len(data)),
        random_state=RANDOM_STATE,
    )

    grid = sns.pairplot(
        sample,
        diag_kind="hist",
        plot_kws={"alpha": 0.20, "s": 12, "edgecolor": "none"},
        diag_kws={"bins": 30, "color": "#4C78A8"},
    )
    grid.figure.suptitle(
        "Pairwise Scatter Plots of Selected Spotify Variables",
        y=1.02,
        fontsize=15,
    )
    grid.figure.savefig(
        FIGURE_DIR / "01_pairwise_scatterplots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(grid.figure)


def create_correlation_heatmap(correlation: pd.DataFrame) -> None:
    """Create a heatmap for all selected quantitative variables."""
    plt.figure(figsize=(11, 8))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Correlation Heatmap of Selected Quantitative Variables", pad=12)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "02_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_gaussian_diagnostics(data: pd.DataFrame) -> None:
    """Create histograms and Q-Q plots for representative variables."""
    qq_sample = data[GAUSSIAN_DIAGNOSTIC_VARIABLES].sample(
        n=min(QQ_SAMPLE_SIZE, len(data)),
        random_state=RANDOM_STATE,
    )

    fig, axes = plt.subplots(
        nrows=len(GAUSSIAN_DIAGNOSTIC_VARIABLES),
        ncols=2,
        figsize=(12, 20),
    )

    for row, variable in enumerate(GAUSSIAN_DIAGNOSTIC_VARIABLES):
        sns.histplot(
            data=data,
            x=variable,
            bins=40,
            kde=True,
            color="#4C78A8",
            ax=axes[row, 0],
        )
        axes[row, 0].set_title(f"Histogram: {variable}")

        stats.probplot(
            qq_sample[variable],
            dist="norm",
            plot=axes[row, 1],
        )
        axes[row, 1].set_title(f"Normal Q-Q Plot: {variable}")

    fig.suptitle(
        "Gaussian Distribution Diagnostics",
        fontsize=16,
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "03_gaussian_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def fit_standardized_regression(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler, sm.regression.linear_model.RegressionResultsWrapper]:
    """Fit popularity on standardized audio feature predictors."""
    scaler = StandardScaler()
    standardized_predictors = pd.DataFrame(
        scaler.fit_transform(data[PREDICTORS]),
        columns=PREDICTORS,
        index=data.index,
    )
    response = data["popularity"]

    model = sm.OLS(
        response,
        sm.add_constant(standardized_predictors, has_constant="add"),
    ).fit()

    return standardized_predictors, response, scaler, model


def model_table(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    """Convert a statsmodels regression result to a compact table."""
    confidence_intervals = model.conf_int()
    table = pd.DataFrame(
        {
            "coefficient": model.params,
            "standard_error": model.bse,
            "t_statistic": model.tvalues,
            "p_value": model.pvalues,
            "ci_2.5%": confidence_intervals[0],
            "ci_97.5%": confidence_intervals[1],
        }
    )
    table.index.name = "variable"
    return table


def backward_elimination(
    predictors: pd.DataFrame,
    response: pd.Series,
    alpha: float = 0.05,
) -> tuple[
    list[str],
    sm.regression.linear_model.RegressionResultsWrapper,
    pd.DataFrame,
]:
    """Remove the predictor with the largest p-value until all p-values <= alpha."""
    selected = list(predictors.columns)
    history: list[dict[str, float | int | str]] = []
    step = 1

    while True:
        model = sm.OLS(
            response,
            sm.add_constant(predictors[selected], has_constant="add"),
        ).fit()
        p_values = model.pvalues.drop("const")
        largest_p_value = float(p_values.max())

        if largest_p_value <= alpha:
            break

        removed_variable = str(p_values.idxmax())
        history.append(
            {
                "step": step,
                "removed_variable": removed_variable,
                "p_value_at_removal": largest_p_value,
            }
        )
        selected.remove(removed_variable)
        step += 1

    history_table = pd.DataFrame(history)
    return selected, model, history_table


def save_regression_results(
    predictors: pd.DataFrame,
    response: pd.Series,
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[
    list[str],
    sm.regression.linear_model.RegressionResultsWrapper,
    pd.DataFrame,
]:
    """Save the full and selected regression results."""
    full_table = model_table(full_model)
    full_table.to_csv(TABLE_DIR / "regression_full_model.csv")
    (TABLE_DIR / "regression_full_model_summary.txt").write_text(
        full_model.summary().as_text(),
        encoding="utf-8",
    )

    selected_variables, selected_model, history = backward_elimination(
        predictors,
        response,
        alpha=0.05,
    )
    model_table(selected_model).to_csv(TABLE_DIR / "regression_selected_model.csv")
    history.to_csv(TABLE_DIR / "backward_elimination_history.csv", index=False)
    (TABLE_DIR / "regression_selected_model_summary.txt").write_text(
        selected_model.summary().as_text(),
        encoding="utf-8",
    )

    return selected_variables, selected_model, history


def orient_pca_axes(
    components: np.ndarray,
    scores: np.ndarray,
    variable_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Choose convenient signs for the first two PCA axes.

    PCA axis signs are arbitrary. The orientation makes positive PC1 correspond
    to higher energy and positive PC2 correspond to higher valence when possible.
    """
    oriented_components = components.copy()
    oriented_scores = scores.copy()

    energy_index = variable_names.index("energy")
    valence_index = variable_names.index("valence")

    if oriented_components[0, energy_index] < 0:
        oriented_components[0, :] *= -1
        oriented_scores[:, 0] *= -1

    if oriented_components[1, valence_index] < 0:
        oriented_components[1, :] *= -1
        oriented_scores[:, 1] *= -1

    return oriented_components, oriented_scores


def run_pca(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize selected variables, fit PCA, and save PCA tables."""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data[SELECTED_VARIABLES])

    pca = PCA()
    raw_scores = pca.fit_transform(standardized_data)
    components, scores = orient_pca_axes(
        pca.components_,
        raw_scores,
        SELECTED_VARIABLES,
    )

    explained_variance = pd.DataFrame(
        {
            "principal_component": [
                f"PC{i}" for i in range(1, len(SELECTED_VARIABLES) + 1)
            ],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    explained_variance.to_csv(
        TABLE_DIR / "pca_explained_variance.csv",
        index=False,
    )

    # For standardized variables, these values are correlations with the PCs.
    correlation_loadings = components.T * np.sqrt(pca.explained_variance_)
    loadings = pd.DataFrame(
        correlation_loadings,
        index=SELECTED_VARIABLES,
        columns=[f"PC{i}" for i in range(1, len(SELECTED_VARIABLES) + 1)],
    )
    loadings.index.name = "variable"
    loadings.to_csv(TABLE_DIR / "pca_correlation_loadings.csv")

    score_table = pd.DataFrame(
        scores[:, :2],
        columns=["PC1", "PC2"],
        index=data.index,
    )
    score_table["acousticness"] = data["acousticness"]
    score_table["track_genre"] = data["track_genre"]

    create_pca_summary_figure(explained_variance, loadings)
    create_pca_projection(score_table)

    return explained_variance, loadings, score_table


def create_pca_summary_figure(
    explained_variance: pd.DataFrame,
    loadings: pd.DataFrame,
) -> None:
    """Create a scree plot and correlation circle in one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    pc_numbers = np.arange(1, len(explained_variance) + 1)
    axes[0].bar(
        pc_numbers,
        explained_variance["explained_variance_ratio"] * 100,
        color="#4C78A8",
        label="Individual",
    )
    axes[0].plot(
        pc_numbers,
        explained_variance["cumulative_explained_variance"] * 100,
        marker="o",
        color="#F58518",
        label="Cumulative",
    )
    axes[0].set_xticks(pc_numbers)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("PCA Explained Variance")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    circle = plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--")
    axes[1].add_artist(circle)
    label_offsets = {
        "danceability": (-0.03, 0.06),
        "valence": (0.03, 0.01),
        "popularity": (-0.06, 0.05),
        "speechiness": (0.05, -0.01),
        "liveness": (0.02, -0.05),
    }
    for variable in loadings.index:
        x = loadings.loc[variable, "PC1"]
        y = loadings.loc[variable, "PC2"]
        axes[1].arrow(
            0,
            0,
            x,
            y,
            color="#4C78A8",
            alpha=0.85,
            head_width=0.025,
            length_includes_head=True,
        )
        x_offset, y_offset = label_offsets.get(variable, (0.0, 0.0))
        axes[1].text(
            x * 1.08 + x_offset,
            y * 1.08 + y_offset,
            variable,
            fontsize=9,
            ha="center",
            va="center",
        )

    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].set_xlim(-1.15, 1.15)
    axes[1].set_ylim(-1.15, 1.15)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("PC1 correlation loading")
    axes[1].set_ylabel("PC2 correlation loading")
    axes[1].set_title("PCA Correlation Circle")
    axes[1].grid(alpha=0.20)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "04_pca_explained_variance_and_correlation_circle.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_pca_projection(score_table: pd.DataFrame) -> None:
    """Create a projection of a random sample onto the first two PCs."""
    projection_sample = score_table.sample(
        n=min(PROJECTION_SAMPLE_SIZE, len(score_table)),
        random_state=RANDOM_STATE,
    )

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        projection_sample["PC1"],
        projection_sample["PC2"],
        c=projection_sample["acousticness"],
        cmap="viridis",
        s=12,
        alpha=0.45,
        edgecolors="none",
    )
    plt.colorbar(scatter, label="Acousticness")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel("First principal component (PC1)")
    plt.ylabel("Second principal component (PC2)")
    plt.title("Projection of Spotify Tracks onto the First Two Principal Components")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "05_pca_projection.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def strongest_correlations(correlation: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the strongest unique correlations by absolute value."""
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    pairs = correlation.where(mask).stack().reset_index()
    pairs.columns = ["variable_1", "variable_2", "correlation"]
    pairs["absolute_correlation"] = pairs["correlation"].abs()
    return pairs.sort_values("absolute_correlation", ascending=False).head(n)


def write_analysis_summary(
    cleaning_summary: dict[str, int],
    descriptive: pd.DataFrame,
    correlation: pd.DataFrame,
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
    selected_variables: list[str],
    selected_model: sm.regression.linear_model.RegressionResultsWrapper,
    elimination_history: pd.DataFrame,
    explained_variance: pd.DataFrame,
    loadings: pd.DataFrame,
) -> None:
    """Write the main numerical results in a compact text summary."""
    top_correlations = strongest_correlations(correlation, n=10)
    top_correlations.to_csv(TABLE_DIR / "strongest_correlations.csv", index=False)

    lines = [
        "SPOTIFY MULTIVARIATE ANALYSIS SUMMARY",
        "",
        "DATA CLEANING",
        f"Original rows: {cleaning_summary['original_rows']}",
        f"Duplicate track IDs removed: {cleaning_summary['duplicate_track_ids']}",
        f"Missing selected values: {cleaning_summary['missing_selected_values']}",
        (
            "Nonpositive-duration rows removed: "
            f"{cleaning_summary['nonpositive_duration_rows']}"
        ),
        f"Rows used in analysis: {cleaning_summary['analysis_rows']}",
        f"Genre labels in original data: {cleaning_summary['genre_count']}",
        "",
        "SELECTED VARIABLE MEANS",
        descriptive["mean"].round(4).to_string(),
        "",
        "STRONGEST CORRELATIONS",
        top_correlations.round(4).to_string(index=False),
        "",
        "FULL STANDARDIZED REGRESSION",
        f"R-squared: {full_model.rsquared:.6f}",
        f"Adjusted R-squared: {full_model.rsquared_adj:.6f}",
        model_table(full_model)[["coefficient", "p_value"]].round(6).to_string(),
        "",
        "BACKWARD ELIMINATION",
        (
            elimination_history.to_string(index=False)
            if not elimination_history.empty
            else "No variables were removed at alpha = 0.05."
        ),
        f"Selected variables: {', '.join(selected_variables)}",
        f"Selected model R-squared: {selected_model.rsquared:.6f}",
        f"Selected model adjusted R-squared: {selected_model.rsquared_adj:.6f}",
        "",
        "PCA EXPLAINED VARIANCE",
        explained_variance.head(6).round(6).to_string(index=False),
        "",
        "PCA CORRELATION LOADINGS FOR PC1 AND PC2",
        loadings[["PC1", "PC2"]].round(4).to_string(),
        "",
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete analysis."""
    prepare_output_directories()
    sns.set_theme(style="whitegrid", context="notebook")

    data, cleaning_summary = load_and_clean_data()
    descriptive, correlation = save_descriptive_tables(data)

    create_pairplot(data)
    create_correlation_heatmap(correlation)
    create_gaussian_diagnostics(data)

    predictors, response, _, full_model = fit_standardized_regression(data)
    selected_variables, selected_model, elimination_history = save_regression_results(
        predictors,
        response,
        full_model,
    )

    explained_variance, loadings, _ = run_pca(data)

    write_analysis_summary(
        cleaning_summary,
        descriptive,
        correlation,
        full_model,
        selected_variables,
        selected_model,
        elimination_history,
        explained_variance,
        loadings,
    )

    print(f"Analysis complete. Results saved to: {OUTPUT_DIR}")
    print(f"Rows used in analysis: {cleaning_summary['analysis_rows']}")
    print(f"Full regression R-squared: {full_model.rsquared:.4f}")
    print(f"Selected variables: {', '.join(selected_variables)}")
    print(
        "PC1 and PC2 cumulative explained variance: "
        f"{explained_variance.loc[1, 'cumulative_explained_variance']:.4f}"
    )


if __name__ == "__main__":
    main()
