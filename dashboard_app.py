import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import ast
import os
from glob import glob
import subprocess
import time
from io import BytesIO

st.set_page_config(page_title='ProjectP Dashboard', layout='wide')

# --- Caching for file loading ---
@st.cache_data(show_spinner=False)
def load_csv(path, nrows=None):
    return pd.read_csv(path, nrows=nrows) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def list_files(pattern):
    return sorted([os.path.basename(f) for f in glob(pattern)])

# --- Sidebar Navigation ---
st.sidebar.title('ProjectP Dashboard')
section = st.sidebar.radio('เลือกหมวดข้อมูล', [
    'Metrics', 'Backtest', 'Feature Importance', 'SHAP', 'Trade Log', 'Compare Runs', 'Custom Analytics', 'Upload/Export', 'Run Pipeline', 'About']
)

# --- Helper: Reload button ---
def reload_button():
    if st.button('🔄 Refresh/Reload Data'):
        st.experimental_rerun()

# --- Helper: Notification ---
def notify(msg, success=True):
    if success:
        st.success(msg)
    else:
        st.error(msg)

# --- Metrics Tab ---
if section == 'Metrics':
    st.title('Metrics Summary')
    reload_button()
    metrics_files = list_files('output_default/metrics_summary_*.csv')
    metrics_file = st.selectbox('เลือกไฟล์ Metrics', metrics_files, index=0 if metrics_files else None)
    metrics_df = None
    if metrics_file:
        metrics_path = os.path.join('output_default', metrics_file)
        metrics_df = load_csv(metrics_path)
        cols = st.columns(6)
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
        for i, m in enumerate(metrics):
            val = metrics_df[m].iloc[0] if m in metrics_df.columns else None
            if pd.notnull(val):
                cols[i].metric(m.upper(), f"{val:.4f}")
            else:
                cols[i].metric(m.upper(), '-')
        with st.expander('Show full metrics table'):
            st.dataframe(metrics_df)
        # Confusion Matrix
        if 'confusion_matrix' in metrics_df.columns:
            st.subheader('Confusion Matrix')
            try:
                cm = ast.literal_eval(metrics_df['confusion_matrix'].iloc[0])
                cm = np.array(cm)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                  labels=dict(x="Predicted", y="Actual", color="Count"))
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.info(f'Confusion matrix parsing error: {e}')
    else:
        st.warning('ไม่พบ metrics summary')

# --- Backtest Tab ---
elif section == 'Backtest':
    st.title('Backtest Visualization')
    reload_button()
    backtest_files = list_files('output_default/backtest_result*.csv')
    backtest_file = st.selectbox('เลือกไฟล์ Backtest', backtest_files, index=0 if backtest_files else None)
    if backtest_file:
        backtest_path = os.path.join('output_default', backtest_file)
        nrows = st.slider('เลือกจำนวนแถวข้อมูล Backtest ที่จะแสดง', 100, 5000, 1000, step=100)
        price_df = load_csv(backtest_path, nrows=nrows)
        # Equity curve
        st.subheader('Equity Curve')
        if 'Equity' in price_df.columns:
            fig_eq = px.line(price_df, x='Time', y='Equity', title='Equity Curve')
            st.plotly_chart(fig_eq, use_container_width=True)
        # Drawdown (ถ้ามี)
        if 'Drawdown' in price_df.columns:
            fig_dd = px.line(price_df, x='Time', y='Drawdown', title='Drawdown')
            st.plotly_chart(fig_dd, use_container_width=True)
        # Candlestick chart
        fig_candle = go.Figure(data=[go.Candlestick(x=price_df['Time'],
            open=price_df['Open'], high=price_df['High'], low=price_df['Low'], close=price_df['Close'])])
        fig_candle.update_layout(title='Backtest OHLC', xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)
        # Volume chart
        fig_vol = px.bar(price_df, x='Time', y='Volume', title='Backtest Volume')
        st.plotly_chart(fig_vol, use_container_width=True)
        # Close price line
        fig_close = px.line(price_df, x='Time', y='Close', title='Backtest Close Price')
        st.plotly_chart(fig_close, use_container_width=True)
        with st.expander('Show raw backtest data'):
            st.dataframe(price_df)
        st.download_button('Download Backtest CSV', data=price_df.to_csv(index=False), file_name=backtest_file)
    else:
        st.warning('ไม่พบ backtest result')

# --- Feature Importance Tab ---
elif section == 'Feature Importance':
    st.title('Feature Importance')
    reload_button()
    fi_files = list_files('output_default/feature_importance_*.csv')
    fi_file = st.selectbox('เลือกไฟล์ Feature Importance', fi_files, index=0 if fi_files else None)
    if fi_file:
        fi_path = os.path.join('output_default', fi_file)
        fi_df = load_csv(fi_path)
        st.dataframe(fi_df)
        fig_fi = px.bar(fi_df, x=fi_df.columns[0], y=fi_df.columns[1], title='Feature Importance')
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.warning('ไม่พบข้อมูล feature importance')

# --- SHAP Tab ---
elif section == 'SHAP':
    st.title('SHAP Explainability')
    reload_button()
    shap_imgs = list_files('output_default/shap_*.png')
    shap_csvs = list_files('output_default/shap_*.csv')
    if shap_imgs:
        for img in shap_imgs:
            st.image(os.path.join('output_default', img), caption=img)
    if shap_csvs:
        for csvf in shap_csvs:
            shap_df = load_csv(os.path.join('output_default', csvf))
            st.dataframe(shap_df)
    if not (shap_imgs or shap_csvs):
        st.warning('ไม่พบข้อมูล SHAP')

# --- Trade Log Tab ---
elif section == 'Trade Log':
    st.title('Trade Log (Walkforward/Realistic)')
    reload_button()
    trade_files = list_files('output_default/trade_log_*.csv')
    trade_file = st.selectbox('เลือกไฟล์ Trade Log', trade_files, index=0 if trade_files else None)
    if trade_file:
        trade_path = os.path.join('output_default', trade_file)
        nrows = st.slider('เลือกจำนวนแถว Trade Log ที่จะแสดง', 100, 10000, 1000, step=100)
        trade_df = load_csv(trade_path, nrows=nrows)
        # Interactive filter
        filter_col = st.selectbox('Filter by column', trade_df.columns)
        filter_val = st.text_input('Filter value (partial match)', '')
        if filter_val:
            trade_df = trade_df[trade_df[filter_col].astype(str).str.contains(filter_val)]
        st.dataframe(trade_df)
        st.download_button('Download Trade Log CSV', data=trade_df.to_csv(index=False), file_name=trade_file)
    else:
        st.warning('ไม่พบ trade log')

# --- Compare Runs Tab ---
elif section == 'Compare Runs':
    st.title('Compare Experiments/Runs')
    reload_button()
    metrics_files = list_files('output_default/metrics_summary_*.csv')
    if len(metrics_files) >= 2:
        files_to_compare = st.multiselect('เลือกไฟล์ Metrics เพื่อเปรียบเทียบ', metrics_files, default=metrics_files[:2])
        if len(files_to_compare) >= 2:
            dfs = [load_csv(os.path.join('output_default', f)).assign(run=f) for f in files_to_compare]
            df_all = pd.concat(dfs)
            st.dataframe(df_all)
            for m in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']:
                if m in df_all.columns:
                    fig = px.bar(df_all, x='run', y=m, title=f'Compare {m.upper()}')
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('ต้องมี metrics summary อย่างน้อย 2 ไฟล์เพื่อเปรียบเทียบ')

# --- Custom Analytics Tab ---
elif section == 'Custom Analytics':
    st.title('Custom Analytics & Visualization')
    st.info('เลือกไฟล์, คอลัมน์, และสูตร/กราฟที่ต้องการวิเคราะห์ได้เอง')
    # เลือกไฟล์
    all_csvs = sorted([os.path.basename(f) for f in glob('output_default/*.csv')])
    file_selected = st.selectbox('เลือกไฟล์ข้อมูล', all_csvs, index=0 if all_csvs else None)
    if file_selected:
        df = load_csv(os.path.join('output_default', file_selected))
        st.dataframe(df.head(100))
        # เลือกคอลัมน์
        cols = df.columns.tolist()
        col_x = st.selectbox('X axis', cols, index=0)
        col_y = st.selectbox('Y axis', cols, index=1 if len(cols)>1 else 0)
        chart_type = st.selectbox('Chart Type', ['Line', 'Bar', 'Scatter', 'Histogram', 'Box', 'Violin', 'Correlation Heatmap'])
        fig = None
        if chart_type == 'Line':
            fig = px.line(df, x=col_x, y=col_y, title=f'Line: {col_y} vs {col_x}')
        elif chart_type == 'Bar':
            fig = px.bar(df, x=col_x, y=col_y, title=f'Bar: {col_y} vs {col_x}')
        elif chart_type == 'Scatter':
            fig = px.scatter(df, x=col_x, y=col_y, title=f'Scatter: {col_y} vs {col_x}')
        elif chart_type == 'Histogram':
            fig = px.histogram(df, x=col_x, title=f'Histogram: {col_x}')
        elif chart_type == 'Box':
            fig = px.box(df, y=col_y, x=col_x, title=f'Box: {col_y} by {col_x}')
        elif chart_type == 'Violin':
            fig = px.violin(df, y=col_y, x=col_x, box=True, points='all', title=f'Violin: {col_y} by {col_x}')
        elif chart_type == 'Correlation Heatmap':
            corr = df.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title='Correlation Heatmap')
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            # Export buttons
            img_bytes = fig.to_image(format="png") if hasattr(fig, 'to_image') else None
            if img_bytes:
                st.download_button('Download Chart as PNG', data=img_bytes, file_name='custom_chart.png', mime='image/png')
            excel_bytes = BytesIO()
            df.to_excel(excel_bytes, index=False)
            st.download_button('Download Data as Excel', data=excel_bytes.getvalue(), file_name='custom_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            st.download_button('Download Data as CSV', data=df.to_csv(index=False), file_name='custom_data.csv')
        # Custom formula
        st.markdown('---')
        st.subheader('Custom Formula')
        formula = st.text_input('ใส่สูตร pandas เช่น df["Close"] - df["Open"]', '')
        if formula:
            try:
                result = eval(formula, {'df': df, 'np': np, 'pd': pd})
                st.write('ผลลัพธ์สูตร:', result.head(10) if hasattr(result, 'head') else result)
            except Exception as e:
                st.error(f'สูตรผิดพลาด: {e}')
    else:
        st.warning('ไม่พบไฟล์ข้อมูลใน output_default/')

# --- Upload/Export Tab ---
elif section == 'Upload/Export':
    st.title('อัปโหลดไฟล์ผลลัพธ์ใหม่ (CSV)')
    uploaded_file = st.file_uploader('เลือกไฟล์ CSV เพื่ออัปเดตผลลัพธ์', type=['csv'])
    if uploaded_file is not None:
        save_path = os.path.join('output_default', uploaded_file.name)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.read())
        st.success(f'อัปโหลดไฟล์ {uploaded_file.name} สำเร็จ! กรุณารีเฟรชหน้าเพื่อดูผลลัพธ์ใหม่')
    st.info('คุณสามารถดาวน์โหลดข้อมูลจากแต่ละหมวดได้ในแต่ละ tab')

# --- Run Pipeline Tab ---
elif section == 'Run Pipeline':
    st.title('Run Full ML Pipeline')
    st.info('คุณสามารถรัน pipeline หรือ backtest ได้จากหน้านี้')
    run_type = st.radio('เลือก pipeline ที่ต้องการรัน', ['Run Full ML Pipeline', 'Run Full ML Pipeline Until AUC>=70'])
    if st.button('🚀 Run Selected Pipeline'):
        with st.spinner('กำลังรัน pipeline... กรุณารอสักครู่'):
            try:
                if run_type == 'Run Full ML Pipeline':
                    result = subprocess.run(['python', 'ProjectP.py', '--run_full_pipeline'], capture_output=True, text=True, timeout=3600)
                else:
                    result = subprocess.run(["bash", "-c", "while true; do OUTPUT=$(python ProjectP.py --run_full_pipeline); echo \"$OUTPUT\"; AUC=$(echo \"$OUTPUT\" | grep -Po 'AUC: \\K[0-9]+\\.?[0-9]*'); if (( $(echo \"$AUC >= 70\" | bc -l) )); then echo \"Success: AUC=$AUC\"; break; else echo \"AUC $AUC < 70, retrying...\"; fi; done"], capture_output=True, text=True, timeout=14400)
                # Clear cache after pipeline run
                load_csv.clear()
                list_files.clear()
                if result.returncode == 0:
                    notify('Pipeline run เสร็จสมบูรณ์! (ข้อมูลล่าสุดพร้อมใช้งาน)')
                    st.text_area('Pipeline Output', result.stdout, height=300)
                else:
                    notify('Pipeline run ล้มเหลว!', success=False)
                    st.text_area('Pipeline Error', result.stderr, height=300)
            except Exception as e:
                notify(f'เกิดข้อผิดพลาด: {e}', success=False)
    st.warning('หลังรัน pipeline ระบบจะ refresh cache อัตโนมัติ หากยังไม่เห็นข้อมูลใหม่ให้กด refresh อีกครั้ง')

# --- About Tab ---
else:
    st.title('About ProjectP Dashboard')
    st.markdown('''
    - **ระบบนี้ออกแบบมาเพื่อวิเคราะห์ผล ML/Quantitative Trading Pipeline อย่างมืออาชีพ**
    - รองรับการแสดงผล metrics, backtest, feature importance, SHAP, trade log, compare runs, และอัปโหลดไฟล์ใหม่
    - UI สวยงาม ใช้งานง่าย รองรับการ refresh/reload ข้อมูลแบบ realtime
    - หากต้องการเพิ่มฟีเจอร์ใหม่ แจ้งทีมงานได้ทันที!
    ''')
