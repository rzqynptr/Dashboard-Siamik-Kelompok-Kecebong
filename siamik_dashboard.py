import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap

# -----------------------------
# Page config & CSS
# -----------------------------
st.set_page_config(
    page_title="SIAMIK Student Portal Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 18px; }
    .stTabs [data-baseweb="tab"] { height:46px; padding-left:16px; padding-right:16px; background:#f3f6fb; border-radius:8px; }
    .stTabs [data-baseweb="tab"]:hover { background:#e8eefb; }
    .stTabs [aria-selected="true"] { background:#1e3a8a; color:white; }
    /* Metric card */
    .metric-card { padding:16px; border-radius:12px; color:white; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
    .muted { color: #6b7280; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Data loader + preparation
# -----------------------------
@st.cache_data
def load_transformed(path="data_final_transformed (1).csv"):
    """Load transformed dataset (used for most analyses)."""
    try:
        df_raw = pd.read_csv(path)
    except Exception as e:
        st.warning(f"⚠️ Failed to load transformed dataset ({path}). Error: {e}")
        df_raw = pd.DataFrame()
    return df_raw

@st.cache_data
def load_raw(path="Data_Responden.csv"):
    """Load raw dataset (used for quick statistics in sidebar)."""
    try:
        raw = pd.read_csv(path)
    except Exception as e:
        st.warning(f"⚠️ Failed to load raw dataset ({path}). Error: {e}")
        raw = pd.DataFrame()
    return raw

def prepare_transformed(df_raw):
    """
    Clean & standardize column names, detect one-hot faculties/prodi, detect problem & priority one-hot groups.
    Returns (df, meta)
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), {'faculty_cols': [], 'prodi_cols': [], 'problem_cols': [], 'priority_cols': [], 'renamed': {}}

    df = df_raw.copy()

    # --- Friendly column mapping (if present in your data) ---
    rename_map = {
        'std_Seberapa mudah Anda mengakses SIAMIK saat war KRS berlangsung?': 'ease_of_access_std',
        'std_Berapa lama rata-rata waktu login yang Anda alami saat war KRS? (.....menit)': 'login_duration_std',
        'std_Seberapa sering Anda mengalami gagal masuk atau error saat login di SIAMIK?': 'login_errors_std',
        'std_Berapa lama waktu yang Anda perlukan untuk menunggu ACC dari sistem/webnya? (....menit/jam)': 'acc_wait_std',
        'std_Seberapa puas Anda secara keseluruhan terhadap SIAMIK dalam proses War KRS dan ACC?': 'overall_satisfaction_std',
        'std_Berdasarkan pengalaman terakhir anda, Bagaimana penilaian anda terhadap kualitas SIAMIK saat war KRS dan ACC?': 'system_quality_std',
        'log_Berapa lama rata-rata waktu login yang Anda alami saat war KRS? (.....menit)': 'login_duration_log',
        'log_Berapa lama waktu yang Anda perlukan untuk menunggu ACC dari sistem/webnya? (....menit/jam)': 'acc_wait_log',
        'lbl_Apakah Anda pernah kehilangan mata kuliah karena slot penuh akibat lambatnya SIAMIK?': 'lost_courses_lbl'
    }

    # rename only existing columns
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # detect one-hot faculty and prodi columns in dataset
    faculty_cols = [c for c in df.columns if c.startswith("Fakultas_") or c.lower().startswith('fakultas_')]
    prodi_cols = [c for c in df.columns if c.startswith("Prodi_") or c.lower().startswith('prodi_')]

    # fallback if not present: try to find singular faculty/prodi columns
    if faculty_cols:
        # get faculty name from the one-hot
        try:
            df['faculty'] = df[faculty_cols].idxmax(axis=1).str.replace('Fakultas_', '', regex=False)
        except Exception:
            # idxmax might fail if booleans aren't numeric; attempt alternative
            df['faculty'] = df[faculty_cols].apply(lambda row: next((c.replace('Fakultas_', '') for c, v in row.items() if bool(v)), 'Unknown'), axis=1)
    else:
        possible = [c for c in df.columns if 'fakultas' in c.lower() or 'faculty' in c.lower()]
        df['faculty'] = df[possible[0]] if possible else 'Unknown'

    if prodi_cols:
        try:
            df['study_program'] = df[prodi_cols].idxmax(axis=1).str.replace('Prodi_', '', regex=False)
        except Exception:
            df['study_program'] = df[prodi_cols].apply(lambda row: next((c.replace('Prodi_', '') for c, v in row.items() if bool(v)), 'Unknown'), axis=1)
    else:
        possible_prodi = [c for c in df.columns if 'prodi' in c.lower() or 'study' in c.lower()]
        df['study_program'] = df[possible_prodi[0]] if possible_prodi else 'Unknown'

    # detect problem and priority one-hot groups
    problem_cols = [c for c in df.columns if c.startswith('Masalah utama') or 'masalah utama' in c.lower()]
    priority_cols = [c for c in df.columns if c.startswith('Jika diberikan kesempatan') or c.startswith('Jika diberikan') or 'prioritas' in c.lower()]

    # create readable 'main_problem' (first chosen problem) if problem_cols exist
    if problem_cols:
        try:
            df['main_problem'] = df[problem_cols].idxmax(axis=1).str.replace('Masalah utama yang paling sering Anda alami saat war KRS?_', '', regex=False)
        except Exception:
            # fallback: pick first True column per row
            def pick_problem(row):
                for c in problem_cols:
                    try:
                        if row.get(c) in [1, True, '1', 'True', 'TRUE', 'true']:
                            return c.replace('Masalah utama yang paling sering Anda alami saat war KRS?_', '')
                    except Exception:
                        continue
                return None
            df['main_problem'] = df.apply(pick_problem, axis=1)
    else:
        df['main_problem'] = None

    # create readable 'improvement_priority' if priority_cols exist
    if priority_cols:
        try:
            df['improvement_priority'] = df[priority_cols].idxmax(axis=1).str.replace('Jika diberikan kesempatan memilih, aspek apa yang paling prioritas untuk diperbaiki pada SIAMIK?_', '', regex=False)
        except Exception:
            def pick_priority(row):
                for c in priority_cols:
                    try:
                        if row.get(c) in [1, True, '1', 'True', 'TRUE', 'true']:
                            return c.replace('Jika diberikan kesempatan memilih, aspek apa yang paling prioritas untuk diperbaiki pada SIAMIK?_', '')
                    except Exception:
                        continue
                return None
            df['improvement_priority'] = df.apply(pick_priority, axis=1)
    else:
        # try common alternative column names
        alt_cols = [c for c in df.columns if 'prioritas' in c.lower() or 'improvement' in c.lower() or 'Jika diberikan' in c]
        if alt_cols:
            df['improvement_priority'] = df[alt_cols[0]]
        else:
            df['improvement_priority'] = None

    # Ensure lost_courses boolean
    if 'lost_courses_lbl' in df.columns:
        try:
            df['lost_courses'] = df['lost_courses_lbl'].astype(bool)
        except Exception:
            df['lost_courses'] = df['lost_courses_lbl'].apply(lambda x: str(x).lower() in ['true', '1', 'yes', 'y']) if 'lost_courses_lbl' in df.columns else False
    elif 'lbl_Apakah Anda pernah kehilangan mata kuliah karena slot penuh akibat lambatnya SIAMIK?' in df.columns:
        try:
            df['lost_courses'] = df['lbl_Apakah Anda pernah kehilangan mata kuliah karena slot penuh akibat lambatnya SIAMIK?'].astype(bool)
        except Exception:
            df['lost_courses'] = df.iloc[:, 0].apply(lambda x: str(x).lower() in ['true', '1', 'yes', 'y'])
    else:
        df['lost_courses'] = False

    meta = {
        'faculty_cols': faculty_cols,
        'prodi_cols': prodi_cols,
        'problem_cols': problem_cols,
        'priority_cols': priority_cols,
        'renamed': rename_map
    }
    return df, meta

# Load datasets
df_raw = load_transformed()
raw_df = load_raw()
df, meta = prepare_transformed(df_raw)

# -----------------------------
# Sidebar: filters and quick stats
# -----------------------------
with st.sidebar:
    st.image("logo_UPN.png", use_container_width=True)
    st.markdown("### 🎯 Filters (interactive)")

    # Build faculty / prodi options from transformed df (so tab filters remain consistent)
    faculties = ["All"] + sorted(df['faculty'].unique().tolist()) if 'faculty' in df.columns else ["All"]
    selected_faculty = st.selectbox("Select Faculty", faculties, index=0)

    prodis = ["All"] + sorted(df['study_program'].unique().tolist()) if 'study_program' in df.columns else ["All"]
    selected_prodi = st.selectbox("Select Study Program (Prodi)", prodis, index=0)

    st.markdown("---")
    st.markdown("### 📊 Quick Statistics")

    # -----------------------------
    # Filter transformed data for download and other UI consistency (kept for compatibility)
    # -----------------------------
    filtered_transformed = df.copy()
    if selected_faculty != "All":
        filtered_transformed = filtered_transformed[filtered_transformed['faculty'] == selected_faculty]
    if selected_prodi != "All":
        filtered_transformed = filtered_transformed[filtered_transformed['study_program'] == selected_prodi]

    # -----------------------------
    # Filter raw_df by the same selected faculty & prodi (if those columns exist in raw)
    # -----------------------------
    filtered_raw = raw_df.copy() if raw_df is not None else pd.DataFrame()
    # raw file sample uses columns "Fakultas" and "Prodi" — handle gracefully
    if not filtered_raw.empty:
        if 'Fakultas' in filtered_raw.columns and selected_faculty != "All":
            # Some transformed faculty values might be trimmed; attempt exact match first, else case-insensitive
            try:
                filtered_raw = filtered_raw[filtered_raw['Fakultas'] == selected_faculty]
            except Exception:
                filtered_raw = filtered_raw[filtered_raw['Fakultas'].str.lower() == str(selected_faculty).lower()]
        if 'Prodi' in filtered_raw.columns and selected_prodi != "All":
            try:
                filtered_raw = filtered_raw[filtered_raw['Prodi'] == selected_prodi]
            except Exception:
                filtered_raw = filtered_raw[filtered_raw['Prodi'].str.lower() == str(selected_prodi).lower()]

    # -----------------------------
    # Raw data column names (expected)
    # -----------------------------
    col_ease = "Seberapa mudah Anda mengakses SIAMIK saat war KRS berlangsung?"
    col_sat = "Seberapa puas Anda secara keseluruhan terhadap SIAMIK dalam proses War KRS dan ACC?"
    col_lost = "Apakah Anda pernah kehilangan mata kuliah karena slot penuh akibat lambatnya SIAMIK?"

    # Convert raw numeric columns to numeric if present (defensive)
    if not filtered_raw.empty:
        for c in [col_ease, col_sat]:
            if c in filtered_raw.columns:
                filtered_raw[c] = pd.to_numeric(filtered_raw[c], errors='coerce')

    # Total respondents (based on raw_df filter)
    total_respondents = len(filtered_raw) if (filtered_raw is not None and not filtered_raw.empty) else 0

    # Compute averages from raw (fallback to transformed if raw not available)
    if col_sat in filtered_raw.columns:
        avg_satisfaction = filtered_raw[col_sat].mean(skipna=True)
    else:
        # fallback to transformed columns (if exist)
        avg_satisfaction = filtered_transformed['overall_satisfaction_std'].mean() if 'overall_satisfaction_std' in filtered_transformed.columns else np.nan

    if col_ease in filtered_raw.columns:
        avg_ease = filtered_raw[col_ease].mean(skipna=True)
    else:
        avg_ease = filtered_transformed['ease_of_access_std'].mean() if 'ease_of_access_std' in filtered_transformed.columns else np.nan

    if col_lost in filtered_raw.columns:
        lost_series = filtered_raw[col_lost].astype(str).str.strip().str.lower()
        lost_pct = (lost_series.isin(['ya', 'yes', 'true', '1'])).mean() * 100
    else:
        lost_pct = filtered_transformed['lost_courses'].mean() * 100 if 'lost_courses' in filtered_transformed.columns else np.nan

    # Metric cards (simple)
    st.markdown(f"""
    <div style="display:flex; gap:10px;">
      <div style="flex:1; background: linear-gradient(135deg,#667eea 0%, #764ba2 100%); padding:12px; border-radius:10px; color:white;">
        <div style="font-size:14px; margin-bottom:6px;">Total Respondents</div>
        <div style="font-size:20px; font-weight:700;">{total_respondents}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.metric("Average Satisfaction", f"{avg_satisfaction:.2f}/5" if not np.isnan(avg_satisfaction) else "N/A")
    st.metric("Avg Ease of Access", f"{avg_ease:.2f}/5" if not np.isnan(avg_ease) else "N/A")
    st.metric("Lost Courses Rate", f"{lost_pct:.1f}%" if not np.isnan(lost_pct) else "N/A")

    st.markdown("---")
    st.markdown("### 🔁 Export / Download")
    # Keep previous behavior: offer download of filtered transformed data
    if st.button("Download filtered CSV"):
        st.download_button("Download CSV", filtered_transformed.to_csv(index=False), file_name="siamik_filtered.csv", mime="text/csv")

    st.markdown("---")
    st.caption("Quick stats use raw dataset (Data_Responden.csv) where available; main analysis uses transformed dataset.")

# -----------------------------
# Main page header
# -----------------------------
st.title("🎓 SIAMIK Student Portal Dashboard")
st.markdown("Transformed survey data analysis — descriptive statistics, performance, problems and improvement priorities.")
st.markdown("Use the left sidebar to filter by Faculty / Study Program. All visuals update dynamically.")

# helper to get filtered df for tabs (based on current sidebar selections)
def get_filtered_df(base_df):
    df_f = base_df.copy()
    if 'faculty' in df_f.columns and selected_faculty != "All":
        df_f = df_f[df_f['faculty'] == selected_faculty]
    if 'study_program' in df_f.columns and selected_prodi != "All":
        df_f = df_f[df_f['study_program'] == selected_prodi]
    return df_f

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_perf, tab_sat, tab_prob, tab_prio = st.tabs([
    "📈 Overview",
    "⚡ System Performance",
    "😊 Satisfaction & Correlation",
    "⚠️ Common Problems",
    "🎯 Improvement Priorities"
])

# -----------------------------
# TAB: Overview (updated version)
# -----------------------------
with tab_overview:
    st.header("📈 Overview — Summary & Key Insights")
    dff = get_filtered_df(df)

    # 🧾 Dataset Description (Bilingual)
    with st.expander("📘 Dataset Description / Deskripsi Dataset", expanded=True):
        st.markdown("""
        ### 🇮🇩 **Deskripsi Dataset**

        Dataset ini berasal dari hasil kuesioner yang disebarkan kepada mahasiswa **UPN Veteran Jawa Timur** 
        untuk mengevaluasi pengalaman mereka menggunakan **SIAMIK** selama proses *war KRS* dan proses *ACC*.  
        Data mencakup kombinasi variabel **numerikal** dan **kategorikal** yang menilai aspek-aspek seperti:
        - Kemudahan login ke sistem  
        - Kecepatan akses dan stabilitas SIAMIK  
        - Lamanya proses ACC melalui sistem/web  
        - Tingkat kepuasan secara keseluruhan terhadap sistem

        Selain itu, mahasiswa juga memberikan tanggapan mengenai kendala yang dialami, seperti:
        - Server yang lambat saat puncak *war KRS*  
        - Slot kelas penuh atau tidak tersedia  
        - Error sistem saat input mata kuliah  
        - Ketidakjelasan notifikasi hasil ACC  

        Tujuan dari pengumpulan data ini adalah untuk **menilai tingkat kepuasan Mahasiswa** serta 
        **mengidentifikasi aspek layanan yang perlu ditingkatkan** agar pengalaman *war KRS* berikutnya lebih efisien dan nyaman.""")



     # ==============================
    # 🎯 Overview Metric Cards (pakai data mentah)
    # ==============================
    raw_filtered = raw_df.copy()

    # Filter sesuai fakultas/prodi yang dipilih
    if selected_faculty != "All" and 'Fakultas' in raw_filtered.columns:
        raw_filtered = raw_filtered[raw_filtered['Fakultas'] == selected_faculty]
    if selected_prodi != "All" and 'Prodi' in raw_filtered.columns:
        raw_filtered = raw_filtered[raw_filtered['Prodi'] == selected_prodi]

    # Hitung metrik dari data mentah
    total_respondents = len(raw_filtered)

    # Lost courses: hitung persentase jawaban "Ya"
    lost_courses_rate = df['lost_courses'].mean() * 100 if 'lost_courses' in df.columns else np.nan

    # Rata-rata waktu login (mentah)
    if 'Berapa lama rata-rata waktu login yang Anda alami saat war KRS? (.....menit)' in raw_filtered.columns:
        avg_login = raw_filtered['Berapa lama rata-rata waktu login yang Anda alami saat war KRS? (.....menit)'].mean()
    else:
        avg_login = np.nan

    # Rata-rata waktu tunggu ACC (mentah)
    if 'Berapa lama waktu yang Anda perlukan untuk menunggu ACC dari sistem/webnya? (....menit/jam)' in raw_filtered.columns:
        avg_acc_wait = raw_filtered['Berapa lama waktu yang Anda perlukan untuk menunggu ACC dari sistem/webnya? (....menit/jam)'].mean()
    else:
        avg_acc_wait = np.nan

    # ==============================
    # 🎨 Tampilan Metric Cards
    # ==============================

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #6366f1, #7e22ce); padding: 25px; border-radius: 15px; color: white;">
            <h3>Total Respondents</h3>
            <h1 style="margin-top: 0;">{total_respondents}</h1>
            <p>Survey participants (filtered)</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ec4899, #f43f5e); padding: 25px; border-radius: 15px; color: white;">
            <h3>Lost Courses</h3>
            <h1 style="margin-top: 0;">{lost_courses_rate:.1f}%</h1>
            <p>Due to full slots</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #06b6d4, #3b82f6); padding: 25px; border-radius: 15px; color: white;">
            <h3>Avg Login (raw)</h3>
            <h1 style="margin-top: 0;">{avg_login:.2f}</h1>
            <p>Minutes (raw data)</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f97316, #facc15); padding: 25px; border-radius: 15px; color: white;">
            <h3>ACC Wait (raw)</h3>
            <h1 style="margin-top: 0;">{avg_acc_wait:.2f}</h1>
            <p>Minutes (raw data)</p>
        </div>
        """, unsafe_allow_html=True)

       # ==============================
    # 📊 Data Preview + Statistics (from RAW data)
    # ==============================
    st.markdown("---")
    st.subheader("🔍 Data Preview (Raw Dataset)")

    # Filter raw_df by selected faculty & prodi
    raw_filtered = raw_df.copy()
    if not raw_filtered.empty:
        if 'Fakultas' in raw_filtered.columns and selected_faculty != "All":
            raw_filtered = raw_filtered[raw_filtered['Fakultas'] == selected_faculty]
        if 'Prodi' in raw_filtered.columns and selected_prodi != "All":
            raw_filtered = raw_filtered[raw_filtered['Prodi'] == selected_prodi]

        # Show first 10 rows
        st.dataframe(raw_filtered.head(10), use_container_width=True)
    else:
        st.info("Raw dataset not available for preview.")

    st.subheader("📊 Descriptive Statistics (Numeric Columns - Transformed Data)")

    if not dff.empty:
    # Pilih hanya kolom numerik dari data hasil transformasi
        numeric_transformed = dff.select_dtypes(include=[np.number])
    
        if not numeric_transformed.empty:
            desc_transformed = numeric_transformed.describe().T.round(3)
            st.dataframe(desc_transformed, use_container_width=True)
        else:
            st.info("No numeric columns found in transformed dataset.")
    else:
        st.info("No transformed data found for descriptive statistics.")


    # ==============================
    # 🥧 Respondent Distribution
    # ==============================
    st.markdown("---")
    st.subheader("👥 Respondent Distribution by Faculty")
    if meta['faculty_cols']:
        fac_counts = {c.replace('Fakultas_', ''): int(df[c].sum()) for c in meta['faculty_cols']}
        fac_df = pd.DataFrame({'Faculty': list(fac_counts.keys()), 'Count': list(fac_counts.values())})
        fig = px.pie(fac_df, values='Count', names='Faculty', title='Respondent Distribution (by Faculty)', hole=0.4)
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    else:
        if 'faculty' in df.columns:
            fac_df = df['faculty'].value_counts().reset_index()
            fac_df.columns = ['Faculty', 'Count']
            fig = px.pie(fac_df, values='Count', names='Faculty', title='Respondent Distribution (by Faculty)', hole=0.4)
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Faculty one-hot columns not found in dataset.")

# -----------------------------
# TAB: System Performance
# -----------------------------
with tab_perf:
    st.header("System Performance Analysis")
    dff = get_filtered_df(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login Duration Distribution")
        # prefer log transformed column if available
        if 'login_duration_log' in dff.columns:
            fig = px.histogram(dff, x='login_duration_log', nbins=25, title='Login Duration (log-transformed)')
            # mean line
            mean_val = dff['login_duration_log'].mean()
            fig.add_vline(x=mean_val, line_dash='dash', annotation_text=f"Mean: {mean_val:.2f}")
            st.plotly_chart(fig, use_container_width=True)
        elif 'login_duration_std' in dff.columns:
            fig = px.histogram(dff, x='login_duration_std', nbins=25, title='Login Duration (standardized)')
            mean_val = dff['login_duration_std'].mean()
            fig.add_vline(x=mean_val, line_dash='dash', annotation_text=f"Mean: {mean_val:.2f}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Login duration column not found.")

    with col2:
        st.subheader("Login Error Frequency")
        if 'login_errors_std' in dff.columns:
            err_counts = dff['login_errors_std'].round(2)
            fig = px.histogram(dff, x='login_errors_std', nbins=10, title='Login Errors (std)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Login errors column not found.")

    st.markdown("---")
    st.subheader("ACC Waiting Time by Faculty (transformed)")
    acc_col = 'acc_wait_log' if 'acc_wait_log' in df.columns else 'acc_wait_std' if 'acc_wait_std' in df.columns else None
    if acc_col and meta['faculty_cols']:
        acc_means = []
        for fcol in meta['faculty_cols']:
            fac_name = fcol.replace('Fakultas_', '')
            mask = df[fcol] == True
            if mask.sum() > 0:
                acc_means.append({'Faculty': fac_name, 'MeanACC': float(df.loc[mask, acc_col].mean()) if pd.api.types.is_numeric_dtype(df.loc[mask, acc_col]) else np.nan})
        acc_df = pd.DataFrame(acc_means).sort_values('MeanACC', ascending=False)
        fig = px.bar(acc_df, x='Faculty', y='MeanACC', title='ACC Wait Time by Faculty (transformed)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ACC waiting column or faculty columns not available for this chart.")

    # Performance metrics table
    st.markdown("---")
    st.subheader("Performance Metrics Summary")
    candidate_metrics = ['ease_of_access_std', 'login_duration_std', 'login_duration_log', 'login_errors_std', 'acc_wait_std', 'acc_wait_log', 'overall_satisfaction_std']
    available = [c for c in candidate_metrics if c in dff.columns]
    perf_rows = []
    for c in available:
        perf_rows.append({
            'Metric': c,
            'Mean': f"{dff[c].mean():.2f}" if pd.api.types.is_numeric_dtype(dff[c]) else 'N/A',
            'Median': f"{dff[c].median():.2f}" if pd.api.types.is_numeric_dtype(dff[c]) else 'N/A',
            'StdDev': f"{dff[c].std():.2f}" if pd.api.types.is_numeric_dtype(dff[c]) else 'N/A'
        })
    if perf_rows:
        st.table(pd.DataFrame(perf_rows))
    else:
        st.info("No performance metrics found for summary table.")

# -----------------------------
# TAB: Satisfaction & Correlation
# -----------------------------
with tab_sat:
    st.header("Satisfaction & Correlation")
    dff = get_filtered_df(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overall Satisfaction Distribution")
        if 'overall_satisfaction_std' in dff.columns:
            fig = px.histogram(dff, x='overall_satisfaction_std', nbins=6, title='Overall Satisfaction (standardized)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Overall satisfaction column not found.")

    with col2:
        st.subheader("Ease of Access Distribution")
        if 'ease_of_access_std' in dff.columns:
            fig = px.histogram(dff, x='ease_of_access_std', nbins=6, title='Ease of Access (standardized)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ease of access column not found.")

    st.markdown("---")
    st.subheader("Correlation Matrix (key numeric variables)")
    # choose numeric cols relevant for correlation
    numeric_cols = [c for c in dff.columns if any(k in c for k in ['_std', '_log', 'ease_of_access', 'overall_satisfaction', 'system_quality'])]
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(dff[c])]
    if numeric_cols:
        corr = dff[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Interpretation tip:** Correlation values close to 1 or -1 indicate strong relationships.")
    else:
        st.info("No numeric columns detected for correlation.")

# -----------------------------
# TAB: Common Problems
# -----------------------------
with tab_prob:
    st.header("Common Problems during War KRS")
    dff = get_filtered_df(df)
    problem_cols = meta['problem_cols']
    if problem_cols:
        # counts per problem
        prob_counts = {}
        try:
            prob_counts = {p.replace('Masalah utama yang paling sering Anda alami saat war KRS?_', ''): int(dff[p].sum()) for p in problem_cols}
            prob_df = pd.DataFrame({'Problem': list(prob_counts.keys()), 'Count': list(prob_counts.values())}).sort_values('Count', ascending=False)
            fig = px.bar(prob_df, x='Count', y='Problem', orientation='h', title='Reported Problems Frequency', color='Count', color_continuous_scale='Reds', text='Count')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Could not compute problem frequencies. Check problem one-hot columns.")

        # severity calculation (frequency * (5 - avg satisfaction))
        st.markdown("### Problem severity (proxy)")
        total_reports = prob_df['Count'].sum() if 'prob_df' in locals() and prob_df['Count'].sum() > 0 else 1
        severity_records = []
        for pcol in problem_cols:
            label = pcol.replace('Masalah utama yang paling sering Anda alami saat war KRS?_', '')
            reporters = dff[dff[pcol] == True] if pcol in dff.columns else pd.DataFrame()
            cnt = int(reporters.shape[0]) if not reporters.empty else 0
            avg_sat_reporters = reporters['overall_satisfaction_std'].mean() if 'overall_satisfaction_std' in reporters.columns and reporters.shape[0] > 0 else np.nan
            severity = (5 - avg_sat_reporters) * (cnt / total_reports) if not np.isnan(avg_sat_reporters) else 0
            severity_records.append({'Problem': label, 'Count': cnt, 'AvgSatisfaction': avg_sat_reporters, 'SeverityScore': severity})
        sev_df = pd.DataFrame(severity_records).sort_values('SeverityScore', ascending=False).reset_index(drop=True)
        if not sev_df.empty:
            st.dataframe(sev_df.round(3), use_container_width=True)
            fig2 = px.scatter(sev_df, x='Count', y='AvgSatisfaction', size='SeverityScore', color='SeverityScore', text='Problem', title='Problem Severity (freq vs avg satisfaction)')
            fig2.update_traces(textposition='top center')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No severity data to show.")

        # distribution by faculty (percentage)
        st.markdown("---")
        st.subheader("Problem distribution by Faculty (%)")
        if meta['faculty_cols']:
            fac_matrix = {}
            for fcol in meta['faculty_cols']:
                fac = fcol.replace('Fakultas_', '')
                fac_matrix[fac] = []
                for pcol in problem_cols:
                    mask = df[fcol] == True
                    pct = df.loc[mask, pcol].mean() * 100 if mask.sum() > 0 else 0.0
                    fac_matrix[fac].append(pct)
            fac_prob_df = pd.DataFrame(fac_matrix, index=[p.replace('Masalah utama yang paling sering Anda alami saat war KRS?_', '') for p in problem_cols])
            fig3 = px.imshow(fac_prob_df, labels=dict(x="Faculty", y="Problem", color="Percentage"), title="Problem Distribution Across Faculties (%)", aspect="auto", color_continuous_scale='YlOrRd')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Faculty one-hot columns not available for this chart.")
    else:
        # fallback: if 'main_problem' exists as a column
        if 'main_problem' in dff.columns:
            prob_df = dff['main_problem'].value_counts().reset_index()
            prob_df.columns = ['Problem', 'Count']
            fig = px.bar(prob_df, x='Count', y='Problem', orientation='h', title='Reported Problems Frequency', color='Count', color_continuous_scale='Reds', text='Count')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No problem columns found in dataset.")

# -----------------------------
# TAB: Improvement Priorities
# -----------------------------
with tab_prio:
    st.header("Improvement Priorities (Student Requests)")
    dff = get_filtered_df(df)
    pr_cols = meta['priority_cols']

    # Build priority counts robustly (use one-hot columns if available, else fallback)
    if pr_cols:
        try:
            pr_counts = {
                p.replace(
                    'Jika diberikan kesempatan memilih, aspek apa yang paling prioritas untuk diperbaiki pada SIAMIK?_',
                    ''
                ): int(dff[p].sum())
                for p in pr_cols
            }
            pr_df = pd.DataFrame({
                'Priority': list(pr_counts.keys()),
                'Count': list(pr_counts.values())
            }).sort_values('Count', ascending=False)
        except Exception:
            records = []
            for p in pr_cols:
                name = p.replace(
                    'Jika diberikan kesempatan memilih, aspek apa yang paling prioritas untuk diperbaiki pada SIAMIK?_',
                    ''
                )
                try:
                    cnt = int(dff[p].astype(bool).sum())
                except Exception:
                    cnt = int(dff[p].sum()) if p in dff.columns else 0
                records.append({'Priority': name, 'Count': cnt})
            pr_df = pd.DataFrame(records).sort_values('Count', ascending=False)
    else:
        # fallback: single-column priority
        if 'improvement_priority' in dff.columns and dff['improvement_priority'].notna().sum() > 0:
            pr_df = dff['improvement_priority'].value_counts().reset_index()
            pr_df.columns = ['Priority', 'Count']
            pr_df = pr_df.sort_values('Count', ascending=False)
        else:
            pr_df = pd.DataFrame(columns=['Priority', 'Count'])

    # ======================================================
    # Jika tidak ada data, tampilkan pesan
    # ======================================================
    if pr_df.empty or pr_df['Count'].sum() == 0:
        st.info("No improvement priority information found.")
    else:
        # ======================================================
        # Siapkan data
        # ======================================================
        sun_df = pr_df.copy()
        sun_df['root'] = 'All'
        total = sun_df['Count'].sum() if sun_df['Count'].sum() > 0 else 1
        sun_df['percentage'] = (sun_df['Count'] / total * 100).round(1)

        # ======================================================
        # Layout dua kolom untuk grafik
        # ======================================================
        col1, col2 = st.columns([1, 1])
        chart_height = 480

        # ======================================================
        # SUNBURST Chart
        # ======================================================
        with col1:
            fig = px.sunburst(
                sun_df,
                path=['root', 'Priority'],
                values='Count',
                title='Improvement Priority Distribution',
                color='Count',
                color_continuous_scale='Blues',
                height=chart_height
            )
            fig.update_traces(
                textinfo='label+percent entry',
                insidetextorientation='radial',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
            )
            st.plotly_chart(fig, use_container_width=True, key="sunburst_chart")

        # ======================================================
        # BAR Chart (Ranking)
        # ======================================================
        with col2:
            fig2 = px.bar(
                sun_df.sort_values('Count', ascending=True),
                x='Count',
                y='Priority',
                orientation='h',
                text='Count',
                title='Top Priorities (Ranking)',
                color='Count',
                color_continuous_scale='Blues',
                height=chart_height
            )
            fig2.update_traces(textposition='outside')
            fig2.update_layout(
                yaxis_title='Priority',
                xaxis_title='Jumlah Responden',
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True, key="bar_chart")

        # ======================================================
        # Tabel & Insight
        # ======================================================
        st.markdown("#### 📊 Rincian Data Improvement Priority")
        sun_df_sorted = sun_df[['Priority', 'Count', 'percentage']].sort_values(by='Count', ascending=False)
        st.dataframe(sun_df_sorted, use_container_width=True)

        # insight utama
        top_priority = sun_df_sorted.iloc[0]['Priority']
        top_percent = sun_df_sorted.iloc[0]['percentage']
        st.success(
            f"🔎 Prioritas utama perbaikan adalah **{top_priority}**, dengan jumlah responden terbanyak (**{top_percent}%** dari total responden)."
        )
         # heatmap: priority preference by faculty
        if meta['faculty_cols']:
            fac_pr_mat = {}
            for fcol in meta['faculty_cols']:
                fac = fcol.replace('Fakultas_', '')
                fac_pr_mat[fac] = []
                for pcol in pr_cols:
                    mask = df[fcol] == True
                    pct = df.loc[mask, pcol].mean() * 100 if mask.sum() > 0 else 0.0
                    fac_pr_mat[fac].append(pct)
            fac_pr_df = pd.DataFrame(fac_pr_mat, index=[p.replace('Jika diberikan kesempatan memilih, aspek apa yang paling prioritas untuk diperbaiki pada SIAMIK?_', '') for p in pr_cols])
            fig3 = px.imshow(fac_pr_df, labels=dict(x="Faculty", y="Priority", color="Percentage"), title="Priority Preferences by Faculty (%)", aspect="auto", color_continuous_scale='YlOrRd')
            st.plotly_chart(fig3, use_container_width=True)
        # ======================================================
        # Action Plan (Top 3 Priorities)
        # ======================================================
        st.markdown("---")
        st.subheader("Recommended Action Plan (Top priorities)")

        top3 = sun_df.sort_values('Count', ascending=False).head(3).reset_index(drop=True)

        if not top3.empty:
            for idx, row in top3.iterrows():
                name = row['Priority']
                cnt = int(row['Count'])
                pct = row['percentage']
                urgency = "🔴 CRITICAL" if idx == 0 else "🟡 HIGH" if idx == 1 else "🟢 MEDIUM"
                st.markdown(f"{urgency} **{name}** — requested by **{pct:.1f}%** of respondents ({cnt} votes)")

                # rekomendasi tindakan kontekstual
                if 'kecepatan' in name.lower() or 'server' in name.lower():
                    st.markdown("- Tingkatkan kapasitas server, optimalkan query, dan tambahkan caching atau CDN.")
                elif 'notifikasi' in name.lower() or 'notification' in name.lower():
                    st.markdown("- Implementasi notifikasi real-time (email/SMS/in-app) dan riwayat notifikasi yang lebih jelas.")
                elif 'proses' in name.lower() or 'approval' in name.lower() or 'acc' in name.lower():
                    st.markdown("- Sederhanakan alur approval, tambahkan SLA, dan buat pengingat otomatis untuk status pengajuan.")
                else:
                    st.markdown("- Lakukan survei kualitatif untuk desain solusi yang lebih tepat, uji coba kecil sebelum penerapan besar.")
        else:
            st.info("No top priorities to show.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:#64748b;'>© 2025 UPNVJT — SIAMIK Dashboard. Developed by Kelompok Kecebong</div>", unsafe_allow_html=True)