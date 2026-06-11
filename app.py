import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

st.set_page_config(page_title="Visualisasi Regresi Linear", layout="wide")

st.title("Visualisasi Regresi Linear Sederhana")
st.markdown("Aplikasi interaktif untuk analisis regresi linear dengan data **x** dan **y**.")

# ---------- SAMPLE DATASETS ----------
SAMPLE_DATA = {
    "Manual": None,
    "Kecil (n=5)": pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 6]}),
    "Sedang (n=10)": pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [2.1, 3.8, 5.2, 6.0, 7.5, 8.1, 9.3, 10.2, 11.0, 12.5]
    }),
    "Dengan Noise": pd.DataFrame({
        "x": np.arange(1, 21),
        "y": 3 * np.arange(1, 21) + 5 + np.random.default_rng(42).normal(0, 5, 20)
    }),
    "Non-linear": pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [1, 1.5, 3, 5, 8, 12, 17, 23, 30, 38]
    }),
}

# ---------- SIDEBAR: INPUT ----------
with st.sidebar:
    st.header("Input Data")

    sample_choice = st.selectbox("Gunakan data contoh", list(SAMPLE_DATA.keys()), index=1)

    if sample_choice == "Manual":
        input_mode = st.radio("Metode input", ["Text", "Upload CSV"], horizontal=True)

        if input_mode == "Text":
            col1, col2 = st.columns(2)
            with col1:
                x_text = st.text_area("Data x", "1, 2, 3, 4, 5", height=100)
            with col2:
                y_text = st.text_area("Data y", "2, 4, 5, 4, 6", height=100)

            if st.button("Proses", type="primary", use_container_width=True):
                try:
                    x = np.array([float(v) for v in x_text.replace(",", " ").split()])
                    y = np.array([float(v) for v in y_text.replace(",", " ").split()])
                except Exception:
                    st.error("Format tidak valid. Gunakan angka pisah koma/spasi.")
                    st.stop()
        else:
            uploaded = st.file_uploader("Upload CSV (kolom: x, y)", type="csv")
            if uploaded is None:
                st.info("Upload file CSV.")
                st.stop()
            df_in = pd.read_csv(uploaded)
            if not {"x", "y"}.issubset(df_in.columns):
                st.error("CSV harus memiliki kolom 'x' dan 'y'.")
                st.stop()
            x, y = df_in["x"].values, df_in["y"].values
    else:
        df_samp = SAMPLE_DATA[sample_choice]
        x, y = df_samp["x"].values, df_samp["y"].values

    if len(x) != len(y):
        st.error("Jumlah x dan y harus sama.")
        st.stop()
    if len(x) < 3:
        st.error("Minimal 3 titik data.")
        st.stop()

# ---------- DATA EDITOR ----------
st.subheader("Edit Data")
df_data = pd.DataFrame({"x": x, "y": y})
df_edited = st.data_editor(df_data, use_container_width=True, num_rows="dynamic",
                            column_config={"x": st.column_config.NumberColumn("x", min_value=-1e12, max_value=1e12),
                                           "y": st.column_config.NumberColumn("y", min_value=-1e12, max_value=1e12)})

x = df_edited["x"].values.astype(float)
y = df_edited["y"].values.astype(float)

if len(x) < 3:
    st.error("Minimal 3 titik data setelah diedit.")
    st.stop()

n = len(x)

# ---------- REGRESI ----------
x_mean, y_mean = np.mean(x), np.mean(y)
Sxx = np.sum((x - x_mean) ** 2)
Sxy = np.sum((x - x_mean) * (y - y_mean))
Syy = np.sum((y - y_mean) ** 2)

beta1 = Sxy / Sxx
beta0 = y_mean - beta1 * x_mean

y_pred = beta0 + beta1 * x
residuals = y - y_pred

SSE = np.sum(residuals ** 2)
SSR = np.sum((y_pred - y_mean) ** 2)
SST = SSR + SSE

R2 = SSR / SST
adj_R2 = 1 - (SSE / (n - 2)) / (SST / (n - 1))

MSE = SSE / (n - 2)
se_beta1 = np.sqrt(MSE / Sxx)
se_beta0 = np.sqrt(MSE * (1 / n + x_mean ** 2 / Sxx))

t_beta1 = beta1 / se_beta1
t_beta0 = beta0 / se_beta0
p_beta1 = 2 * (1 - stats.t.cdf(abs(t_beta1), n - 2))
p_beta0 = 2 * (1 - stats.t.cdf(abs(t_beta0), n - 2))

alpha = 0.05
t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
ci_beta1 = (beta1 - t_crit * se_beta1, beta1 + t_crit * se_beta1)
ci_beta0 = (beta0 - t_crit * se_beta0, beta0 + t_crit * se_beta0)

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Regresi", "Diagnostik", "Prediksi", "Statistik", "Download"])

# ========== TAB 1: REGRESI ==========
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.set_style("whitegrid")

        ax.scatter(x, y, color="#2166ac", s=65, label="Data", zorder=5, edgecolors="white", linewidth=0.5)

        x_sorted = np.sort(x)
        x_grid = np.linspace(x.min(), x.max(), 200)
        y_grid = beta0 + beta1 * x_grid

        # Confidence interval for mean response
        x_mean_arr = x_grid
        se_fit = np.sqrt(MSE * (1 / n + (x_mean_arr - x_mean) ** 2 / Sxx))
        ci_upper = y_grid + t_crit * se_fit
        ci_lower = y_grid - t_crit * se_fit
        ax.fill_between(x_grid, ci_lower, ci_upper, color="#2166ac", alpha=0.12, label="CI (95%)")

        # Prediction interval
        se_pred = np.sqrt(MSE * (1 + 1 / n + (x_mean_arr - x_mean) ** 2 / Sxx))
        pi_upper = y_grid + t_crit * se_pred
        pi_lower = y_grid - t_crit * se_pred
        ax.fill_between(x_grid, pi_lower, pi_upper, color="#b2182b", alpha=0.08, label="PI (95%)")

        ax.plot(x_grid, y_grid, color="#b2182b", linewidth=2.2, label=f"y = {beta0:.3f} + {beta1:.3f}x", zorder=4)

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title("Regresi Linear dengan CI dan PI 95%", fontsize=13)
        ax.legend(fontsize=10, loc="best")

        st.pyplot(fig)

    with col_right:
        st.metric("Persamaan", f"y = {beta0:.4f} + {beta1:.4f}x")
        st.metric("R-squared", f"{R2:.6f}")
        st.metric("Adj. R-squared", f"{adj_R2:.6f}")
        st.metric("Std. Error (MSE)", f"{np.sqrt(MSE):.6f}")

        st.divider()
        st.caption("Koefisien")
        coef_df = pd.DataFrame({
            "Koefisien": ["Intercept", "Slope (x)"],
            "Estimasi": [f"{beta0:.6f}", f"{beta1:.6f}"],
            "Std. Error": [f"{se_beta0:.6f}", f"{se_beta1:.6f}"],
            "t": [f"{t_beta0:.4f}", f"{t_beta1:.4f}"],
            "p-value": [f"{p_beta0:.6f}", f"{p_beta1:.6f}"],
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.caption("Interval Kepercayaan 95%")
        ci_df = pd.DataFrame({
            "Parameter": ["Intercept", "Slope"],
            "CI Bawah": [f"{ci_beta0[0]:.6f}", f"{ci_beta1[0]:.6f}"],
            "CI Atas": [f"{ci_beta0[1]:.6f}", f"{ci_beta1[1]:.6f}"],
        })
        st.dataframe(ci_df, use_container_width=True, hide_index=True)

    # ---------- TABEL DATA ----------
    st.subheader("Tabel Data")
    df_table = pd.DataFrame({
        "x": x,
        "y": y,
        "y_pred": y_pred,
        "Residual": residuals,
    })
    st.dataframe(df_table, use_container_width=True)

# ========== TAB 2: DIAGNOSTIK ==========
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.stem(x, residuals, linefmt="gray", markerfmt="oC0", basefmt="r-")
        ax1.axhline(0, color="red", linestyle="--", linewidth=1)
        ax1.set_xlabel("x")
        ax1.set_ylabel("Residual")
        ax1.set_title("Residual vs x")
        st.pyplot(fig1)

        # Shapiro-Wilk normality test
        if n >= 3 and n <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            sw_norm = "Normal" if shapiro_p > alpha else "Tidak Normal"
        else:
            shapiro_stat = shapiro_p = np.nan
            sw_norm = "N/A"

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.get_lines()[1].set_color("red")
        ax2.get_lines()[1].set_linewidth(1.5)
        ax2.set_title("Q-Q Plot Residual")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        # Fitted vs Residual
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(y_pred, residuals, color="#2166ac", s=50, edgecolors="white", linewidth=0.5)
        ax3.axhline(0, color="red", linestyle="--")
        ax3.set_xlabel("Fitted values")
        ax3.set_ylabel("Residuals")
        ax3.set_title("Fitted vs Residual")
        st.pyplot(fig3)

    with col4:
        st.subheader("Uji Normalitas Residual")
        if n >= 3 and n <= 5000:
            st.metric("Shapiro-Wilk W", f"{shapiro_stat:.4f}")
            st.metric("p-value", f"{shapiro_p:.6f}")
            st.metric("Kesimpulan", sw_norm)
        else:
            st.info("Jumlah data tidak sesuai untuk uji Shapiro-Wilk.")

        st.divider()
        st.subheader("Ringkasan Diagnostik")
        skew = stats.skew(residuals)
        kurt = stats.kurtosis(residuals, fisher=True)
        st.metric("Skewness", f"{skew:.4f}")
        st.metric("Kurtosis (excess)", f"{kurt:.4f}")
        if abs(skew) < 1 and abs(kurt) < 2:
            st.success("Residual mendekati distribusi Normal.")
        else:
            st.warning("Residual menyimpang dari distribusi Normal.")

# ========== TAB 3: PREDIKSI ==========
with tab3:
    st.subheader("Prediksi untuk Nilai x Baru")

    x_min, x_max = float(x.min()), float(x.max())
    x_new = st.number_input("Masukkan nilai x", value=(x_min + x_max) / 2, format="%f")

    if x_new:
        y_new = beta0 + beta1 * x_new
        se_fit_new = np.sqrt(MSE * (1 / n + (x_new - x_mean) ** 2 / Sxx))
        se_pred_new = np.sqrt(MSE * (1 + 1 / n + (x_new - x_mean) ** 2 / Sxx))

        ci_new_low = y_new - t_crit * se_fit_new
        ci_new_high = y_new + t_crit * se_fit_new
        pi_new_low = y_new - t_crit * se_pred_new
        pi_new_high = y_new + t_crit * se_pred_new

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Prediksi y", f"{y_new:.6f}")
        col_b.metric("CI 95% (rata-rata)", f"({ci_new_low:.4f}, {ci_new_high:.4f})")
        col_c.metric("PI 95% (individu)", f"({pi_new_low:.4f}, {pi_new_high:.4f})")
        col_d.metric("Std. Error Prediksi", f"{se_pred_new:.6f}")

        # Show prediction on plot
        fig_p, ax_p = plt.subplots(figsize=(8, 4.5))
        sns.set_style("whitegrid")
        ax_p.scatter(x, y, color="#2166ac", s=55, label="Data", zorder=5, edgecolors="white", linewidth=0.5)

        x_grid_p = np.linspace(x.min(), x.max(), 200)
        y_grid_p = beta0 + beta1 * x_grid_p
        se_grid_p = np.sqrt(MSE * (1 / n + (x_grid_p - x_mean) ** 2 / Sxx))
        ax_p.plot(x_grid_p, y_grid_p, color="#b2182b", linewidth=2, label="Regresi", zorder=4)
        ax_p.fill_between(x_grid_p, y_grid_p - t_crit * se_grid_p, y_grid_p + t_crit * se_grid_p,
                          color="#2166ac", alpha=0.12)

        ax_p.scatter([x_new], [y_new], color="green", s=120, zorder=6, marker="D",
                     label=f"Prediksi ({x_new:.2f}, {y_new:.2f})", edgecolors="darkgreen", linewidth=1.5)
        ax_p.legend(fontsize=10)
        ax_p.set_xlabel("x")
        ax_p.set_ylabel("y")
        ax_p.set_title(f"Prediksi untuk x = {x_new:.4f}")
        st.pyplot(fig_p)

# ========== TAB 4: STATISTIK ==========
with tab4:
    st.subheader("Analisis Varians (ANOVA)")

    df_reg = 1
    df_res = n - 2
    df_tot = n - 1

    MSR = SSR / df_reg
    F_stat = MSR / MSE
    F_pval = 1 - stats.f.cdf(F_stat, df_reg, df_res)

    anova = pd.DataFrame({
        "Sumber": ["Regresi", "Residual", "Total"],
        "df": [df_reg, df_res, df_tot],
        "SS": [f"{SSR:.6f}", f"{SSE:.6f}", f"{SST:.6f}"],
        "MS": [f"{MSR:.6f}", f"{MSE:.6f}", ""],
        "F": [f"{F_stat:.6f}" if F_stat else "", "", ""],
        "p-value": [f"{F_pval:.6f}" if F_pval else "", "", ""],
    })
    st.dataframe(anova, use_container_width=True, hide_index=True)

    st.divider()

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Korelasi Pearson (r)", f"{np.corrcoef(x, y)[0, 1]:.6f}")
    col_s2.metric("Koefisien Determinasi (R²)", f"{R2:.6f}")
    col_s3.metric("Adj. R²", f"{adj_R2:.6f}")

    st.divider()

    st.subheader("Statistik Deskriptif")
    desc = pd.DataFrame({
        "": ["x", "y"],
        "n": [n, n],
        "Mean": [f"{x_mean:.4f}", f"{y_mean:.4f}"],
        "Std": [f"{np.std(x, ddof=1):.4f}", f"{np.std(y, ddof=1):.4f}"],
        "Min": [f"{x.min():.4f}", f"{y.min():.4f}"],
        "Max": [f"{x.max():.4f}", f"{y.max():.4f}"],
    })
    st.dataframe(desc, use_container_width=True, hide_index=True)

# ========== TAB 5: DOWNLOAD ==========
with tab5:
    st.subheader("Download Hasil")

    # CSV hasil
    df_out = pd.DataFrame({
        "x": x, "y": y, "y_pred": y_pred, "residual": residuals
    })
    csv_buf = io.BytesIO()
    df_out.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    st.download_button("Download Data + Prediksi (CSV)", data=csv_buf,
                       file_name="regresi_linear.csv", mime="text/csv",
                       type="primary", use_container_width=True)

    # Ringkasan regresi
    summary_lines = [
        "=== HASIL REGRESI LINEAR ===",
        f"n = {n}",
        f"Persamaan: y = {beta0:.6f} + {beta1:.6f}x",
        "",
        "--- Koefisien ---",
        f"Intercept (beta0) = {beta0:.6f}  (SE = {se_beta0:.6f}, t = {t_beta0:.4f}, p = {p_beta0:.6f})",
        f"Slope (beta1)    = {beta1:.6f}  (SE = {se_beta1:.6f}, t = {t_beta1:.4f}, p = {p_beta1:.6f})",
        "",
        "--- ANOVA ---",
        f"SSR = {SSR:.6f}, df = {df_reg}",
        f"SSE = {SSE:.6f}, df = {df_res}",
        f"SST = {SST:.6f}, df = {df_tot}",
        f"F   = {F_stat:.6f}, p = {F_pval:.6f}",
        "",
        f"R-squared   = {R2:.6f}",
        f"Adj R-squared = {adj_R2:.6f}",
        "",
        "--- Interval Kepercayaan 95% ---",
        f"Intercept: ({ci_beta0[0]:.6f}, {ci_beta0[1]:.6f})",
        f"Slope:     ({ci_beta1[0]:.6f}, {ci_beta1[1]:.6f})",
    ]
    summary_text = "\n".join(summary_lines)

    st.download_button("Download Ringkasan (TXT)", data=summary_text,
                       file_name="ringkasan_regresi.txt", mime="text/plain",
                       use_container_width=True)

    # Download plot sebagai PNG
    fig_dl, ax_dl = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax_dl.scatter(x, y, color="#2166ac", s=70, label="Data", zorder=5, edgecolors="white", linewidth=0.5)
    x_grid_dl = np.linspace(x.min(), x.max(), 200)
    y_grid_dl = beta0 + beta1 * x_grid_dl
    se_dl = np.sqrt(MSE * (1 / n + (x_grid_dl - x_mean) ** 2 / Sxx))
    ax_dl.fill_between(x_grid_dl, y_grid_dl - t_crit * se_dl, y_grid_dl + t_crit * se_dl,
                        color="#2166ac", alpha=0.12, label="CI 95%")
    ax_dl.plot(x_grid_dl, y_grid_dl, color="#b2182b", linewidth=2.5, label=f"y = {beta0:.3f} + {beta1:.3f}x")
    ax_dl.set_xlabel("x", fontsize=12)
    ax_dl.set_ylabel("y", fontsize=12)
    ax_dl.set_title("Regresi Linear Sederhana", fontsize=14)
    ax_dl.legend(fontsize=11)

    img_buf = io.BytesIO()
    fig_dl.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
    img_buf.seek(0)
    plt.close(fig_dl)

    st.download_button("Download Plot (PNG)", data=img_buf,
                       file_name="plot_regresi.png", mime="image/png",
                       use_container_width=True)
