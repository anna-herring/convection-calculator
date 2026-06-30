import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["mathtext.default"] = "regular"
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import argparse
import sys
import os

VOXEL_XLSX = None  # resolved at runtime from the CSV directory


def load_voxel_size(csv_path):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    if stem.endswith("_PP"):
        stem = stem[:-3]
    # Look for the xlsx in the same directory as the CSV
    xlsx = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "Samples_voxel_size.xlsx")
    if not os.path.exists(xlsx):
        print(f"  [voxel] Samples_voxel_size.xlsx not found next to CSV, no scaling applied")
        return None
    try:
        df = pd.read_excel(xlsx, engine="openpyxl")
        df.columns = [c.strip() for c in df.columns]
        names = df["Sample_name"].str.strip()
        exact = df[names == stem]
        if not exact.empty:
            return float(exact["voxel_size"].iloc[0])
        best_len, best_val = 0, None
        for _, row in df.iterrows():
            sn = str(row["Sample_name"]).strip()
            common = len(os.path.commonprefix([stem, sn]))
            if common > best_len:
                best_len, best_val = common, float(row["voxel_size"])
        if best_len > 10:
            print(f"  [voxel] fuzzy match (prefix len {best_len}) for: {stem}")
            return best_val
        print(f"  [voxel] WARNING: no match for '{stem}', no scaling applied")
        return None
    except Exception as e:
        print(f"  [voxel] Could not read voxel size file: {e}")
        return None


def compute_landscape(pairs, t_values, n_layers=5, chunk_size=50000):
    n_t = len(t_values)
    if len(pairs) == 0:
        return np.zeros((n_layers, n_t))
    bd = np.array(pairs)
    top_k = np.zeros((n_layers, n_t))
    for start in range(0, len(bd), chunk_size):
        chunk = bd[start:start + chunk_size]
        b = chunk[:, 0:1]
        d = chunk[:, 1:2]
        t = t_values[np.newaxis, :]
        mid = (b + d) / 2.0
        tent = np.where(
            (t <= b) | (t >= d), 0.0,
            np.where(t <= mid, t - b, d - t)
        )
        combined = np.vstack([top_k, tent])
        top_k = np.sort(combined, axis=0)[::-1, :][:n_layers, :]
    return top_k


def load_pairs(csv_path, dimension):
    df = pd.read_csv(
        csv_path,
        comment="#",
        sep=r"\s+",
        header=None,
        names=["birth", "death", "dimension",
               "creator_x", "creator_y", "creator_z",
               "destructor_x", "destructor_y", "destructor_z"],
        na_values=["-", "inf", "-inf"]
    )
    df = df[df["dimension"] == dimension].copy()
    df = df[df["death"].notna() & df["birth"].notna()]
    df = df[df["death"] > df["birth"]]
    pairs = list(zip(df["birth"], df["death"]))
    return pairs, df


def landscape_stats(landscape, t_values):
    stats = {}
    for k, layer in enumerate(landscape):
        if np.any(layer > 0):
            peak_idx = np.argmax(layer)
            stats["lambda_%d" % (k + 1)] = {
                "max_height": float(layer[peak_idx]),
                "peak_t": float(t_values[peak_idx]),
                "L2_norm": float(np.sqrt(np.trapz(layer**2, t_values))),
                "L1_norm": float(np.trapz(layer, t_values)),
            }
    return stats


def identify_exceptional_features(pairs, df_dim, n_sigma=2.0):
    heights = np.array([(d - b) / 2.0 for b, d in pairs])
    mean_h = np.mean(heights)
    std_h = np.std(heights)
    max_sigma = float(np.max((heights - mean_h) / std_h)) if std_h > 0 else 0.0
    exceptional_mask = heights >= mean_h + n_sigma * std_h
    exceptional_df = df_dim[exceptional_mask].copy()
    exceptional_df["persistence"] = exceptional_df["death"] - exceptional_df["birth"]
    exceptional_df["peak_height"] = heights[exceptional_mask]
    exceptional_df["sigmas_above_mean"] = (heights[exceptional_mask] - mean_h) / std_h
    return exceptional_df.sort_values("persistence", ascending=False), max_sigma


def plot_landscape(landscape, t_values, dimension, n_layers,
                   exceptional_pairs=None, output_path=None, title_extra="", x_label="t"):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = mpl_cm.get_cmap("viridis")
    colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    for k in range(n_layers):
        layer = landscape[k]
        if np.any(layer > 0):
            ax.plot(t_values, layer, color=colors[k],
                    linewidth=1.8, label=r"$\lambda_{%d}$" % (k + 1))
            ax.fill_between(t_values, 0, layer, color=colors[k], alpha=0.08)

    if exceptional_pairs is not None and len(exceptional_pairs) > 0:
        bd = np.array(exceptional_pairs)
        b = bd[:, 0:1]
        d = bd[:, 1:2]
        mid = (b + d) / 2.0
        t = t_values[np.newaxis, :]
        tents = np.where(
            (t <= b) | (t >= d), 0.0,
            np.where(t <= mid, t - b, d - t)
        )
        for i, tent_row in enumerate(tents):
            alpha = max(0.15, 0.7 - i * 0.04)
            ax.plot(t_values, tent_row, color="crimson", linewidth=0.9, alpha=alpha,
                    label="exceptional" if i == 0 else "_nolegend_")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(r"$\lambda_k(t)$", fontsize=12)
    title = r"Persistence landscape $-$ $\beta_{%d}$ features" % dimension
    if title_extra:
        title += "\n" + title_extra
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute persistence landscapes from a persistence pairs CSV.")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--layers", type=int, default=5,
                        help="Number of landscape layers (default 5). lambda_1 is the "
                             "most topologically significant; higher layers capture "
                             "progressively less prominent features.")
    parser.add_argument("--resolution", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=None,
                        help="If set, overlay the tent functions of exceptional pairs "
                             "(those exceeding this many sigma above the mean peak height) "
                             "in red on the landscape plot.")
    parser.add_argument("--chunk", type=int, default=50000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    label = os.path.splitext(os.path.basename(args.csv))[0]

    voxel_m = load_voxel_size(args.csv)
    if voxel_m is not None:
        scale = voxel_m * 1e6
        unit_label = "microns"
    else:
        scale = 1.0
        unit_label = "filtration value (unscaled)"

    print(f"Loading b{args.dim} pairs from: {args.csv}")
    pairs, df_dim = load_pairs(args.csv, args.dim)
    print(f"Found {len(pairs)} valid pairs | voxel: {voxel_m} m -> scale x{scale} ({unit_label})")

    if len(pairs) == 0:
        print("No valid pairs found.")
        sys.exit(1)

    scaled_pairs = [(b * scale, d * scale) for b, d in pairs]
    t_min = min(b for b, d in scaled_pairs)
    t_max = max(d for b, d in scaled_pairs)
    t_values = np.linspace(t_min, t_max, args.resolution)
    print(f"t range: [{t_min:.4f}, {t_max:.4f}] {unit_label}")

    landscape = compute_landscape(scaled_pairs, t_values, n_layers=args.layers, chunk_size=args.chunk)

    stats = landscape_stats(landscape, t_values)
    print(f"Landscape statistics (b{args.dim}):")
    for name, s in stats.items():
        print(f"  {name}: max={s['max_height']:.4f} at t={s['peak_t']:.4f} {unit_label}, "
              f"L1={s['L1_norm']:.4f}, L2={s['L2_norm']:.4f}")

    sigma_thresh = args.sigma if args.sigma is not None else 2.0
    exceptional, max_sigma = identify_exceptional_features(scaled_pairs, df_dim, n_sigma=sigma_thresh)
    print(f"Max sigma above mean peak height: {max_sigma:.2f}")

    exceptional_pairs_to_plot = None
    if args.sigma is not None:
        print(f"Exceptional features (>{args.sigma} sigma): {len(exceptional)} found")
        if len(exceptional) > 0:
            cols = ["birth", "death", "persistence", "peak_height", "sigmas_above_mean"]
            spatial = [c for c in df_dim.columns if any(
                tag in c for tag in ["creator", "destructor", "x", "y", "z"])]
            cols = [c for c in cols + spatial if c in exceptional.columns]
            print(exceptional[cols].to_string(index=False))
        exceptional_pairs_to_plot = list(zip(
            exceptional["birth"] * scale, exceptional["death"] * scale))

    if args.plot or args.output:
        plot_landscape(
            landscape, t_values, args.dim, args.layers,
            exceptional_pairs=exceptional_pairs_to_plot,
            output_path=args.output,
            title_extra=label,
            x_label=f"t ({unit_label})"
        )

    return landscape, t_values, exceptional


if __name__ == "__main__":
    main()

