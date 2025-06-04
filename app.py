import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from persiantools.jdatetime import JalaliDate

# ----------- تنظیمات صفحه و پس زمینه و رنگ فونت -----------
st.set_page_config(page_title="داشبورد پایش برق کنسانتره", layout="wide")

page_bg_img = '''
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1581093588401-1ebdd1b6b47d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
    background-size: cover;
    background-attachment: fixed;
    color: white;
    font-family: Tahoma;
    font-size: 16px;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------- آپلود لوگو توسط کاربر -----------
st.sidebar.subheader("🏷️ بارگذاری لوگو شرکت")
uploaded_logo = st.sidebar.file_uploader("آپلود لوگو (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_logo is not None:
    st.sidebar.image(uploaded_logo, width=150)

# ------------ بارگذاری فایل اکسل -------------
uploaded_file = st.file_uploader("📂 لطفاً فایل اکسل کنسانتره را بارگذاری کنید", type=["xlsx"])

def make_unique_columns(cols):
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    return new_cols

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    all_sheets = xls.sheet_names

    dfs = []
    for sheet in all_sheets:
        df_sheet = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)
        raw_headers = df_sheet.iloc[1].fillna("بدون عنوان")
        df_sheet = df_sheet.dropna(axis=1, how="all")
        raw_headers = raw_headers[:df_sheet.shape[1]]

        unique_headers = make_unique_columns(raw_headers)
        df_data = df_sheet[2:]
        df_data.columns = unique_headers
        df_data = df_data.rename(columns={df_data.columns[0]: "تاریخ"})

        df_data["تاریخ"] = pd.to_datetime(df_data["تاریخ"], errors="coerce")
        df_data = df_data.dropna(subset=["تاریخ"])

        for col in df_data.columns:
            if col != "تاریخ":
                df_data[col] = pd.to_numeric(df_data[col], errors="coerce")

        df_data['کارخانه'] = sheet
        dfs.append(df_data)

    df = pd.concat(dfs, ignore_index=True)

    df['تاریخ شمسی'] = df['تاریخ'].apply(lambda x: JalaliDate(x).strftime('%Y/%m/%d') if pd.notnull(x) else "")

    # ---------- فیلتر کارخانه ----------
    factories = df['کارخانه'].unique().tolist()
    selected_factories = st.sidebar.multiselect("🏭 انتخاب کارخانه (کارخانه‌ها)", factories, default=factories)

    filtered_df = df[df['کارخانه'].isin(selected_factories)]

    # ---------- فیلتر بازه زمانی ----------
    st.sidebar.header("🎯 فیلتر بازه زمانی")
    min_date = filtered_df["تاریخ"].min()
    max_date = filtered_df["تاریخ"].max()
    start_date, end_date = st.sidebar.date_input("بازه زمانی", [min_date, max_date])
    st.sidebar.text(f"از تاریخ: {JalaliDate(start_date)} تا تاریخ: {JalaliDate(end_date)}")

    mask = (filtered_df["تاریخ"] >= pd.to_datetime(start_date)) & (filtered_df["تاریخ"] <= pd.to_datetime(end_date))
    filtered_df = filtered_df.loc[mask]

    st.title("📊 داشبورد پایش مصرف برق تجهیزات کنسانتره")

    # استخراج ستون‌های عددی فقط مربوط به کارخانه‌های انتخاب‌شده
    columns = filtered_df.select_dtypes(include="number").columns.tolist()
    columns = [c for c in columns if c not in ['کارخانه']]

    st.subheader("📌 مقایسه میانگین مصرف چند تجهیز")
    selected_columns = st.multiselect("🔌 انتخاب تجهیزات:", columns)

    if selected_columns:
        st.subheader("🔍 شاخص‌های کلیدی (KPI)")
        kpi_cols = st.columns(len(selected_columns))
        for i, col in enumerate(selected_columns):
            with kpi_cols[i]:
                st.metric("میانگین", f"{filtered_df[col].mean():.2f} MWh")
                st.metric("بیشترین", f"{filtered_df[col].max():.2f} MWh")
                st.metric("کمترین", f"{filtered_df[col].min():.2f} MWh")
                st.metric("مجموع سالیانه", f"{filtered_df[col].sum():.2f} MWh")

        mean_values = filtered_df[selected_columns].mean().reset_index()
        mean_values.columns = ["تجهیز", "میانگین مصرف"]

        fig_bar = go.Figure(data=[
            go.Bar(
                x=mean_values["تجهیز"],
                y=mean_values["میانگین مصرف"],
                text=mean_values["میانگین مصرف"].round(2),
                textposition="outside",
                marker=dict(color='skyblue'),
                opacity=0.9
            )
        ])

        fig_bar.update_layout(
            title="📊 مقایسه میانگین مصرف برق تجهیزات",
            template="plotly_dark",
            xaxis_title="تجهیز",
            yaxis_title="میانگین مصرف (MWh)",
            font=dict(family="Tahoma", size=14),
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📈 مشاهده روند مصرف یک تجهیز")
    selected_single = st.selectbox("🧠 انتخاب تجهیز:", columns)

    if selected_single:
        fig_line = px.line(
            filtered_df,
            x="تاریخ",
            y=selected_single,
            title=f"📈 روند مصرف برق تجهیز: {selected_single}",
            template="plotly_white",
            markers=True
        )
        fig_line.update_layout(font=dict(family="Tahoma", size=14), height=500)
        st.plotly_chart(fig_line, use_container_width=True)

    # ---------- نمودار مصرف ماهیانه ----------
    st.subheader("📆 نمودار مصرف ماهیانه تجهیزات")
    filtered_df['ماه شمسی'] = filtered_df['تاریخ'].apply(lambda x: JalaliDate(x).strftime('%Y/%m'))
    monthly_column = st.selectbox("📌 انتخاب تجهیز برای نمودار ماهیانه", columns)

    if monthly_column:
        monthly_df = (
            filtered_df.groupby("ماه شمسی")[monthly_column]
            .sum()
            .reset_index()
            .sort_values("ماه شمسی")
        )

        fig_month = go.Figure(data=[
            go.Bar(
                x=monthly_df["ماه شمسی"],
                y=monthly_df[monthly_column],
                text=monthly_df[monthly_column].round(2),
                textposition="outside",
                marker_color='salmon',
                opacity=0.85
            )
        ])

        fig_month.update_layout(
            title=f"📊 مجموع مصرف ماهیانه تجهیز: {monthly_column}",
            xaxis_tickangle=-45,
            template="plotly_white",
            yaxis_title="مصرف (MWh)",
            font=dict(family="Tahoma", size=14),
            height=500
        )
        st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("📋 جدول داده‌های فیلتر شده (شامل تاریخ شمسی)")
    display_df = filtered_df.copy()
    cols = display_df.columns.tolist()
    if 'تاریخ شمسی' in cols:
        cols.insert(0, cols.pop(cols.index('تاریخ شمسی')))
    display_df = display_df[cols]
    st.dataframe(display_df, use_container_width=True)

    # ---------- دانلود فایل ----------
    st.subheader("📥 دانلود فایل خروجی")
    to_excel = display_df.copy()
    to_excel['تاریخ'] = to_excel['تاریخ'].dt.strftime('%Y-%m-%d')

    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        to_excel.to_excel(writer, index=False, sheet_name='گزارش')

    st.download_button(
        label="📤 دانلود خروجی به اکسل",
        data=output.getvalue(),
        file_name="گزارش_پایش.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("لطفاً ابتدا فایل اکسل کنسانتره را بارگذاری کنید.")
