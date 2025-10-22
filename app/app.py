import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="🌌 Aging Theory Universe Explorer")

# Загружаем данные
df = pd.read_csv('aging_theories_universe.csv')

# Цветовая карта для Q1
color_map = {
    'Yes, quantitatively shown': 'blue',
    'Yes, but not shown': 'orange',
    'No': 'red',
    'Error': 'gray',
    'No text available': 'lightgray'
}

df['color'] = df['q1'].map(color_map).fillna('gray')

# 🌐 Главная визуализация — Scatter Plot по годам и кластерам
st.header("🌐 The Aging Theory Universe")
st.write("Each dot = a paper. Color = Q1 answer. Size = cluster size. Hover to see details.")

fig = px.scatter(df,
                 x='theory_id',
                 y='paper_year',
                 color='color',
                 size='cluster_id',
                 hover_data=['paper_name', 'paper_url', 'source', 'q1', 'q2', 'q3'],
                 title="Theories by Year & Biomarker Status (Q1)",
                 labels={'color': 'Q1 Status'})
st.plotly_chart(fig, use_container_width=True)

# 📊 Heatmap Q-Answers
st.header("📊 Q-Answer Heatmap")
q_cols = [f'q{i}' for i in range(1,10)]
heatmap_data = df[q_cols].apply(lambda x: x.map({'Yes': 1, 'No': 0, 'Yes, quantitatively shown': 2, 'Yes, but not shown': 1.5, 'No text available': 0}), axis=0)
fig_heat = px.density_heatmap(heatmap_data, x=q_cols, y=df.index, title="Theory Answer Patterns Across 1000+ Papers")
st.plotly_chart(fig_heat, use_container_width=True)

# 📋 Таблица с фильтрами
st.header("📋 Filtered Papers")
q1_filter = st.selectbox("Filter by Q1:", ["All"] + list(df['q1'].unique()))
if q1_filter != "All":
    filtered_df = df[df['q1'] == q1_filter]
else:
    filtered_df = df

st.dataframe(filtered_df[['source', 'paper_name', 'paper_year', 'paper_url', 'q1', 'q2', 'q3', 'q4']], height=400)

# 🗺️ Cluster Map
st.header("🗺️ Cluster Visualization")
fig_cluster = px.scatter(df, x='cluster_id', y='paper_year', color='cluster_id', hover_data=['paper_name', 'q1'])
st.plotly_chart(fig_cluster, use_container_width=True)

# 💡 Информация
st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"Total papers: {len(df)}")
st.sidebar.write(f"Clusters found: {len(set(df['cluster_id']))}")
st.sidebar.write(f"Sources: {df['source'].nunique()} sources")
st.sidebar.write("Powered by ChatGPT-4o 🤖")
