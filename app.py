import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon India Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Dark industrial theme */
.stApp {
    background: radial-gradient(circle at 20% 20%, #1a1d27 0%, #0d0f14 100%);
    color: #e8e8e8;
}
.stSidebar {
    background: rgba(17, 19, 24, 0.7) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* KPI Cards with Glassmorphism */
.kpi-card {
    background: rgba(26, 29, 39, 0.4);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.kpi-card:hover {
    border-color: #ff6b35;
    transform: translateY(-4px);
    background: rgba(26, 29, 39, 0.6);
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b35, #ff9e7d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8a8f9e;
    margin-top: 0.6rem;
    font-weight: 600;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #6a6f7e;
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #e8e8e8;
    margin: 2.5rem 0 1.2rem 0;
    display: flex;
    align-items: center;
}
.section-header::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 24px;
    background: #ff6b35;
    margin-right: 12px;
    border-radius: 2px;
}

.insight-box {
    background: rgba(19, 22, 31, 0.6);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 4px solid #4ade80;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #c8cdd8;
    line-height: 1.5;
}
.insight-box b { color: #4ade80; font-weight: 600; }
.warn-box { border-left-color: #f59e0b; }
.warn-box b { color: #f59e0b; }

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #1e2129; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2a2d3a; }

/* Dataframe styling */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING & CLEANING ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/amazon.csv")

    def clean_price(x):
        if pd.isna(x): return np.nan
        return float(re.sub(r'[₹,]', '', str(x)).strip())

    df['discounted_price'] = df['discounted_price'].apply(clean_price)
    df['actual_price'] = df['actual_price'].apply(clean_price)
    df['discount_percentage'] = df['discount_percentage'].apply(
        lambda x: float(str(x).replace('%', '').strip()) if pd.notna(x) else np.nan)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = df['rating_count'].apply(
        lambda x: float(str(x).replace(',', '').strip()) if pd.notna(x) else 0)
    df['main_category'] = df['category'].apply(lambda x: str(x).split('|')[0])
    df['sub_category'] = df['category'].apply(
        lambda x: str(x).split('|')[1] if len(str(x).split('|')) > 1 else '')
    df['savings'] = df['actual_price'] - df['discounted_price']
    df['price_bucket'] = pd.cut(
        df['discounted_price'],
        bins=[0, 200, 500, 1000, 5000, 200000],
        labels=['Budget\n<₹200', 'Economy\n₹200-500', 'Mid\n₹500-1K',
                'Premium\n₹1K-5K', 'Luxury\n₹5K+']
    )
    df['rating_tier'] = pd.cut(df['rating'], bins=[0, 3, 3.5, 4, 4.5, 5.01],
                                labels=['Poor', 'Below Avg', 'Average', 'Good', 'Excellent'])
    df['popularity_score'] = (df['rating'] * np.log1p(df['rating_count'])).round(2)
    df['brand'] = df['product_name'].apply(lambda x: str(x).split(' ')[0])
    df['sentiment_proxy'] = (df['rating'] * np.log10(df['rating_count'] + 1) / 5).clip(0, 5).round(2)
    return df

df = load_data()
COLORS = ['#ff6b35', '#4ade80', '#60a5fa', '#f59e0b', '#e879f9',
          '#fb7185', '#34d399', '#a78bfa', '#38bdf8', '#fbbf24']
PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c8cdd8', family='DM Sans'),
    xaxis=dict(gridcolor='#1e2129', showline=False),
    yaxis=dict(gridcolor='#1e2129', showline=False),
    colorway=COLORS,
    margin=dict(l=10, r=10, t=40, b=10)
)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Syne; font-size: 1.4rem; font-weight: 800; color: #ff6b35; margin-bottom: 0.2rem;'>
    🛒 Amazon India
    </div>
    <div style='font-size: 0.75rem; color: #7a7f8e; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1.5rem;'>
    Product Intelligence
    </div>
    """, unsafe_allow_html=True)

    categories = ['All'] + sorted(df['main_category'].unique().tolist())
    selected_cat = st.selectbox("📦 Category", categories)

    price_range = st.slider(
        "💰 Discounted Price (₹)",
        int(df['discounted_price'].min()),
        int(df['discounted_price'].max()),
        (int(df['discounted_price'].min()), int(df['discounted_price'].max()))
    )
    min_rating = st.slider("⭐ Minimum Rating", 1.0, 5.0, 3.0, 0.1)
    min_reviews = st.number_input("📝 Min Review Count", 0, 100000, 100, 100)

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.72rem; color: #7a7f8e; text-transform: uppercase; letter-spacing: 0.08em;'>
    Navigation
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("Navigation", [
        "📊 Overview",
        "💰 Pricing",
        "⭐ Ratings",
        "🏷️ Discounts",
        "🔍 Explorer",
        "🏢 Brand Hub",
        "📈 Business Insights"
    ], label_visibility="collapsed")

# ─── FILTER DATA ─────────────────────────────────────────────────────────────
fdf = df.copy()
if selected_cat != 'All':
    fdf = fdf[fdf['main_category'] == selected_cat]
fdf = fdf[
    (fdf['discounted_price'] >= price_range[0]) &
    (fdf['discounted_price'] <= price_range[1]) &
    (fdf['rating'] >= min_rating) &
    (fdf['rating_count'] >= min_reviews)
]

# ─── KPI BAR (always visible) ────────────────────────────────────────────────
st.markdown(f"""
<div style='font-family: Syne; font-size: 1.6rem; font-weight: 800; color: #e8e8e8; margin-bottom: 0.2rem;'>
Amazon India · Product Analytics
</div>
<div style='font-size: 0.8rem; color: #7a7f8e; margin-bottom: 1.2rem;'>
{len(fdf):,} products matching filters &nbsp;·&nbsp; {fdf['main_category'].nunique()} categories
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, f"₹{fdf['discounted_price'].median():,.0f}", "Median Price", f"Avg ₹{fdf['discounted_price'].mean():,.0f}", "🏷️"),
    (c2, f"{fdf['rating'].mean():.2f} ★", "Avg Rating", f"Across {len(fdf):,} products", "⭐"),
    (c3, f"{fdf['discount_percentage'].mean():.1f}%", "Avg Discount", f"Max {fdf['discount_percentage'].max():.0f}%", "📉"),
    (c4, f"₹{fdf['savings'].mean():,.0f}", "Avg Savings", f"Total ₹{fdf['savings'].sum()/1e6:.1f}M", "💰"),
    (c5, f"{fdf['rating_count'].sum()/1e6:.1f}M", "Total Reviews", f"Avg {fdf['rating_count'].mean():,.0f}/product", "💬"),
]
for col, val, label, sub, icon in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="section-header">Category Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        cat_counts = fdf['main_category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Products']
        fig = px.bar(
            cat_counts, x='Products', y='Category',
            orientation='h',
            labels={'Products': 'Number of Products', 'Category': ''},
            color='Products',
            color_continuous_scale=['#1e2129', '#ff6b35'],
            title="Products per Category"
        )
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False,
                          title_font=dict(size=14, color='#e8e8e8'))
        fig.update_traces(hovertemplate='<b>%{y}</b><br>%{x} products<extra></extra>')
        st.plotly_chart(fig, width="stretch")

    with col2:
        sub_counts = fdf['sub_category'].value_counts().head(12).reset_index()
        sub_counts.columns = ['Sub-Category', 'Products']
        fig2 = px.pie(
            sub_counts, values='Products', names='Sub-Category',
            title="Sub-Category Share (Top 12)",
            hole=0.55,
            color_discrete_sequence=COLORS
        )
        fig2.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'),
                           legend=dict(font=dict(size=9)))
        fig2.update_traces(textposition='outside', textfont_size=9,
                           hovertemplate='<b>%{label}</b><br>%{value} products (%{percent})<extra></extra>')
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-header">Price Segment Distribution</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        bucket_counts = fdf['price_bucket'].value_counts().sort_index().reset_index()
        bucket_counts.columns = ['Segment', 'Products']
        fig3 = px.bar(
            bucket_counts, x='Segment', y='Products',
            labels={'Segment': 'Price Segment', 'Products': 'Count'},
            color='Products',
            color_continuous_scale=['#1e2129', '#4ade80'],
            title="Products per Price Segment"
        )
        fig3.update_layout(**PLOT_THEME, coloraxis_showscale=False,
                           title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig3, width="stretch")

    with col4:
        fig4 = px.histogram(
            fdf.dropna(subset=['discounted_price']),
            x='discounted_price', nbins=50,
            labels={'discounted_price': 'Discounted Price (₹)'},
            title="Price Distribution",
            color_discrete_sequence=['#ff6b35']
        )
        fig4.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig4, width="stretch")

    # Quick insights
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    top_cat = fdf['main_category'].value_counts().idxmax()
    budget_pct = (fdf['price_bucket'] == 'Budget\n<₹200').sum() / len(fdf) * 100 if len(fdf) > 0 else 0
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        st.markdown(f'<div class="insight-box"><b>Dominant Category:</b> {top_cat} holds {fdf["main_category"].value_counts().iloc[0]} products ({fdf["main_category"].value_counts().iloc[0]/len(fdf)*100:.1f}% of catalog)</div>', unsafe_allow_html=True)
    with ci2:
        st.markdown(f'<div class="insight-box"><b>Budget-Friendly:</b> {budget_pct:.1f}% of products are priced under ₹200, making the catalog accessible to mass market</div>', unsafe_allow_html=True)
    with ci3:
        st.markdown(f'<div class="insight-box warn-box"><b>Price Skew:</b> Median ₹{fdf["discounted_price"].median():,.0f} vs Mean ₹{fdf["discounted_price"].mean():,.0f} — heavy right-tail from luxury products</div>', unsafe_allow_html=True)

# PAGE: PRICING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💰 Pricing":
    st.markdown('<div class="section-header">Price vs Rating Relationship</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            fdf.dropna(subset=['discounted_price', 'rating']),
            x='discounted_price', y='rating',
            size='rating_count', size_max=30,
            color='main_category',
            hover_name='product_name',
            log_x=True,
            title="Price vs Rating (bubble = review count)",
            labels={'discounted_price': 'Discounted Price ₹ (log)', 'rating': 'Rating'},
            color_discrete_sequence=COLORS
        )
        fig.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig, width="stretch")

    with col2:
        cat_price = fdf.groupby('main_category').agg(
            avg_price=('discounted_price', 'mean'),
            avg_actual=('actual_price', 'mean')
        ).round(0).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Actual Price', x=cat_price['main_category'],
                               y=cat_price['avg_actual'], marker_color='#2a2d3a'))
        fig2.add_trace(go.Bar(name='Discounted Price', x=cat_price['main_category'],
                               y=cat_price['avg_price'], marker_color='#ff6b35'))
        fig2.update_layout(**PLOT_THEME, barmode='overlay',
                           title='Actual vs Discounted Price by Category',
                           title_font=dict(size=14, color='#e8e8e8'),
                           legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-header">Pricing by Segment</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(
            fdf.dropna(subset=['discounted_price', 'main_category']),
            x='main_category', y='discounted_price',
            color='main_category',
            title="Price Spread per Category",
            labels={'discounted_price': 'Discounted Price ₹', 'main_category': ''},
            color_discrete_sequence=COLORS,
            log_y=True
        )
        fig3.update_layout(**PLOT_THEME, showlegend=False,
                           title_font=dict(size=14, color='#e8e8e8'))
        fig3.update_xaxes(tickangle=-20)
        st.plotly_chart(fig3, width="stretch")

    with col4:
        seg_stats = fdf.groupby('price_bucket', observed=True).agg(
            count=('product_id', 'count'),
            avg_rating=('rating', 'mean'),
            avg_discount=('discount_percentage', 'mean')
        ).reset_index()
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Bar(x=seg_stats['price_bucket'], y=seg_stats['count'],
                               name='# Products', marker_color='#60a5fa'), secondary_y=False)
        fig4.add_trace(go.Scatter(x=seg_stats['price_bucket'], y=seg_stats['avg_rating'],
                                   name='Avg Rating', marker_color='#ff6b35',
                                   mode='lines+markers', line=dict(width=3)), secondary_y=True)
        fig4.update_layout(**PLOT_THEME, title='Segment: Count & Avg Rating',
                           title_font=dict(size=14, color='#e8e8e8'),
                           legend=dict(orientation='h', x=0.5, xanchor='center', y=1.1))
        fig4.update_yaxes(title_text="Products", secondary_y=False, gridcolor='#1e2129')
        fig4.update_yaxes(title_text="Rating", secondary_y=True, showgrid=False)
        st.plotly_chart(fig4, width="stretch")

    # Top 10 most expensive vs cheapest
    st.markdown('<div class="section-header">Extreme Value Products</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        top10 = fdf.nlargest(10, 'discounted_price')[['product_name', 'discounted_price', 'rating', 'main_category']]
        top10['product_name'] = top10['product_name'].str[:50] + '...'
        top10.columns = ['Product', 'Price (₹)', 'Rating', 'Category']
        st.markdown("**🔝 Top 10 Most Expensive**")
        st.dataframe(top10, width="stretch", hide_index=True)
    with c6:
        bot10 = fdf.nsmallest(10, 'discounted_price')[['product_name', 'discounted_price', 'rating', 'main_category']]
        bot10['product_name'] = bot10['product_name'].str[:50] + '...'
        bot10.columns = ['Product', 'Price (₹)', 'Rating', 'Category']
        st.markdown("**💸 Top 10 Budget Picks**")
        st.dataframe(bot10, width="stretch", hide_index=True)

# PAGE: RATINGS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⭐ Ratings":
    st.markdown('<div class="section-header">Rating Landscape</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(fdf.dropna(subset=['rating']), x='rating', nbins=30,
                           title="Rating Distribution",
                           color_discrete_sequence=['#4ade80'],
                           labels={'rating': 'Rating', 'count': 'Products'})
        fig.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig, width="stretch")

    with col2:
        tier_counts = fdf['rating_tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Products']
        fig2 = px.pie(tier_counts, values='Products', names='Tier',
                      title="Rating Tiers", hole=0.55,
                      color_discrete_sequence=COLORS)
        fig2.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig2, width="stretch")

    with col3:
        fig3 = px.scatter(fdf.dropna(subset=['rating', 'rating_count']),
                          x='rating_count', y='rating',
                          color='main_category',
                          log_x=True,
                          title="Rating vs Review Count",
                          labels={'rating_count': 'Review Count (log)', 'rating': 'Rating'},
                          color_discrete_sequence=COLORS)
        fig3.update_layout(**PLOT_THEME, showlegend=False,
                           title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig3, width="stretch")

    st.markdown('<div class="section-header">Category Rating Performance</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)

    with col4:
        cat_rating = fdf.groupby('main_category').agg(
            avg_rating=('rating', 'mean'),
            total_reviews=('rating_count', 'sum'),
            products=('product_id', 'count')
        ).reset_index().sort_values('avg_rating', ascending=True)
        fig4 = px.bar(cat_rating, x='avg_rating', y='main_category',
                      orientation='h', color='avg_rating',
                      color_continuous_scale=['#dc2626', '#f59e0b', '#4ade80'],
                      title="Average Rating by Category",
                      labels={'avg_rating': 'Avg Rating', 'main_category': ''})
        fig4.update_layout(**PLOT_THEME, coloraxis_showscale=False,
                           title_font=dict(size=14, color='#e8e8e8'))
        fig4.add_vline(x=4.0, line_dash='dash', line_color='#ff6b35',
                       annotation_text='4.0 threshold', annotation_font_color='#ff6b35')
        st.plotly_chart(fig4, width="stretch")

    with col5:
        top_products = fdf.nlargest(15, 'popularity_score')[
            ['product_name', 'rating', 'rating_count', 'popularity_score', 'discounted_price']].copy()
        top_products['product_name'] = top_products['product_name'].str[:45] + '...'
        fig5 = px.bar(top_products.sort_values('popularity_score'),
                      x='popularity_score', y='product_name',
                      orientation='h', color='rating',
                      color_continuous_scale=['#dc2626', '#4ade80'],
                      title="Top 15 Products by Popularity Score",
                      labels={'popularity_score': 'Score (rating × log reviews)', 'product_name': ''})
        fig5.update_layout(**PLOT_THEME, coloraxis_showscale=False,
                           title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig5, width="stretch")

    st.markdown('<div class="section-header">Underrated Gems — High Discount, High Rating, Low Visibility</div>', unsafe_allow_html=True)
    gems = fdf[
        (fdf['rating'] >= 4.2) &
        (fdf['discount_percentage'] >= 40) &
        (fdf['rating_count'] < fdf['rating_count'].quantile(0.4))
    ].sort_values('discount_percentage', ascending=False).head(10)
    gems_display = gems[['product_name', 'discounted_price', 'rating', 'discount_percentage', 'rating_count']].copy()
    gems_display['product_name'] = gems_display['product_name'].str[:60]
    gems_display.columns = ['Product', 'Price ₹', 'Rating', 'Discount %', 'Reviews']
    st.dataframe(gems_display, width="stretch", hide_index=True)

# PAGE: DISCOUNTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏷️ Discounts":
    st.markdown('<div class="section-header">Discount Landscape</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(fdf.dropna(subset=['discount_percentage']),
                           x='discount_percentage', nbins=30,
                           title="Discount % Distribution",
                           color_discrete_sequence=['#f59e0b'],
                           labels={'discount_percentage': 'Discount %'})
        fig.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig, width="stretch")

    with col2:
        cat_disc = fdf.groupby('main_category')['discount_percentage'].agg(
            ['mean', 'max', 'min']).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Avg Discount', x=cat_disc['main_category'],
                               y=cat_disc['mean'], marker_color='#f59e0b'))
        fig2.add_trace(go.Scatter(name='Max Discount', x=cat_disc['main_category'],
                                   y=cat_disc['max'], mode='markers',
                                   marker=dict(color='#ff6b35', size=10, symbol='diamond')))
        fig2.update_layout(**PLOT_THEME, title='Discount by Category',
                           title_font=dict(size=14, color='#e8e8e8'),
                           legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-header">Discount vs Price vs Rating Triangle</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.scatter(
            fdf.dropna(subset=['discount_percentage', 'discounted_price', 'rating']),
            x='discount_percentage', y='discounted_price',
            color='rating', size='rating_count', size_max=25,
            color_continuous_scale=['#dc2626', '#f59e0b', '#4ade80'],
            title="Discount % vs Price (color=rating)",
            labels={'discount_percentage': 'Discount %', 'discounted_price': 'Price ₹'},
            log_y=True
        )
        fig3.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig3, width="stretch")

    with col4:
        # Savings heatmap by category and price segment
        heatmap_data = fdf.groupby(['main_category', 'price_bucket'], observed=True)['savings'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='main_category', columns='price_bucket', values='savings').fillna(0)
        fig4 = px.imshow(heatmap_pivot,
                         color_continuous_scale=['#0d0f14', '#ff6b35'],
                         title="Avg Savings: Category × Price Segment",
                         labels=dict(color="Avg Savings ₹"))
        fig4.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig4, width="stretch")

    # Best discount deals
    st.markdown('<div class="section-header">🔥 Hottest Deals — Max Discount + High Rating</div>', unsafe_allow_html=True)
    deals = fdf[(fdf['rating'] >= 4.0)].sort_values('discount_percentage', ascending=False).head(15)
    deals_display = deals[['product_name', 'actual_price', 'discounted_price',
                            'discount_percentage', 'savings', 'rating']].copy()
    deals_display['product_name'] = deals_display['product_name'].str[:55]
    deals_display.columns = ['Product', 'MRP ₹', 'Sale Price ₹', 'Discount %', 'You Save ₹', 'Rating']
    st.dataframe(deals_display, width="stretch", hide_index=True)

    # Suspicious discounts
    st.markdown('<div class="section-header">⚠️ Inflated MRP Alert — Very High Discounts (Potential Fake MRP)</div>', unsafe_allow_html=True)
    suspicious = fdf[fdf['discount_percentage'] >= 75].sort_values('discount_percentage', ascending=False)
    st.markdown(f'<div class="insight-box warn-box"><b>{len(suspicious)} products</b> have discounts ≥ 75%. These may have inflated MRPs — worth scrutiny before purchase.</div>', unsafe_allow_html=True)
    sus_display = suspicious[['product_name', 'actual_price', 'discounted_price', 'discount_percentage', 'rating']].head(10).copy()
    sus_display['product_name'] = sus_display['product_name'].str[:55]
    sus_display.columns = ['Product', 'MRP ₹', 'Sale ₹', 'Discount %', 'Rating']
    st.dataframe(sus_display, width="stretch", hide_index=True)

# PAGE: EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explorer":
    st.markdown('<div class="section-header">Search & Explore Products</div>', unsafe_allow_html=True)

    search = st.text_input("🔍 Search by product name or keyword", placeholder="e.g. cable, earphone, lamp...")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sort_by = st.selectbox("Sort by", ['popularity_score', 'rating', 'discounted_price', 'discount_percentage', 'rating_count'])
    with col_f2:
        sort_dir = st.radio("Order", ['Descending', 'Ascending'], horizontal=True)
    with col_f3:
        n_results = st.slider("Show results", 10, 100, 25, 5)

    results = fdf.copy()
    if search:
        results = results[results['product_name'].str.contains(search, case=False, na=False)]

    results = results.sort_values(sort_by, ascending=(sort_dir == 'Ascending')).head(n_results)

    st.markdown(f"**{len(results)} products found**")

    display_cols = ['product_name', 'main_category', 'discounted_price',
                    'actual_price', 'discount_percentage', 'rating', 'rating_count', 'savings']
    res_display = results[display_cols].copy()
    res_display['product_name'] = res_display['product_name'].str[:60]
    res_display.columns = ['Product', 'Category', 'Price ₹', 'MRP ₹', 'Disc %', 'Rating', 'Reviews', 'Savings ₹']

    st.dataframe(res_display, width="stretch", hide_index=True)

    # Visual Gallery
    st.markdown('<div class="section-header">Product Gallery</div>', unsafe_allow_html=True)
    gallery_count = min(len(results), 16)
    for i in range(0, gallery_count, 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < gallery_count:
                prod = results.iloc[i + j]
                with col:
                    st.markdown(f"""
                    <div style="border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 10px; background: rgba(255,255,255,0.03); height: 100%;">
                        <img src="{prod['img_link']}" style="width: 100%; border-radius: 5px; margin-bottom: 10px; height: 150px; object-fit: contain;">
                        <div style="font-size: 0.85rem; font-weight: 600; color: #e8e8e8; margin-bottom: 5px; height: 3.4em; overflow: hidden;">{prod['product_name']}</div>
                        <div style="color: #ff6b35; font-weight: 700;">₹{prod['discounted_price']:,.0f} <span style="font-size: 0.7rem; color: #7a7f8e; font-weight: 400; text-decoration: line-through;">₹{prod['actual_price']:,.0f}</span></div>
                        <div style="font-size: 0.8rem; color: #4ade80;">⭐ {prod['rating']} ({prod['rating_count']:,.0f})</div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if len(results) > 0:
        st.markdown('<div class="section-header">Result Analytics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(results.dropna(subset=['discounted_price', 'rating']),
                             x='discounted_price', y='rating',
                             color='main_category', size='rating_count', size_max=30,
                             hover_name='product_name',
                             title=f"Price vs Rating for '{search or 'All'}' results",
                             color_discrete_sequence=COLORS)
            fig.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig2 = px.histogram(results.dropna(subset=['discount_percentage']),
                                x='discount_percentage',
                                title="Discount Distribution in Results",
                                color_discrete_sequence=['#60a5fa'])
            fig2.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
            st.plotly_chart(fig2, width="stretch")

# PAGE: BRAND HUB
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏢 Brand Hub":
    st.markdown('<div class="section-header">Brand Performance Analytics</div>', unsafe_allow_html=True)

    brand_stats = fdf.groupby('brand').agg(
        products=('product_id', 'count'),
        avg_rating=('rating', 'mean'),
        avg_discount=('discount_percentage', 'mean'),
        avg_price=('discounted_price', 'mean'),
        total_reviews=('rating_count', 'sum'),
        sentiment=('sentiment_proxy', 'mean')
    ).reset_index()

    # Filter for brands with significant presence
    min_b_products = 1 if len(fdf) < 50 else 3
    brand_stats = brand_stats[brand_stats['products'] >= min_b_products].sort_values('products', ascending=False)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = px.scatter(
            brand_stats.head(30),
            x='avg_price', y='avg_rating',
            size='products', color='sentiment',
            hover_name='brand',
            text='brand',
            log_x=True,
            title="Brand Position: Price vs Rating (Size=Volume, Color=Sentiment)",
            color_continuous_scale='RdYlGn',
            labels={'avg_price': 'Avg Price ₹ (log)', 'avg_rating': 'Avg Rating'}
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(**PLOT_THEME, height=500)
        st.plotly_chart(fig, width="stretch")

    with col2:
        top_brands = brand_stats.head(15).sort_values('total_reviews', ascending=True)
        fig2 = px.bar(
            top_brands, x='total_reviews', y='brand',
            orientation='h',
            title="Top 15 Brands by Market Reach (Total Reviews)",
            color='avg_rating',
            color_continuous_scale='Viridis',
            labels={'total_reviews': 'Total Reviews', 'brand': ''}
        )
        fig2.update_layout(**PLOT_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-header">Brand Value Quadrant</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([1, 1.5])
    with c3:
        st.markdown("""
        <div class="insight-box">
        <b>💡 Quadrant Logic:</b><br>
        • <b>Premium Leaders:</b> High Rating, High Price<br>
        • <b>Value Kings:</b> High Rating, Low Price<br>
        • <b>Niche/Budget:</b> Low Rating, Low Price<br>
        • <b>Overpriced:</b> Low Rating, High Price
        </div>
        """, unsafe_allow_html=True)

        avg_r = brand_stats['avg_rating'].median()
        avg_p = brand_stats['avg_price'].median()

        brand_stats['quadrant'] = 'Other'
        brand_stats.loc[(brand_stats['avg_rating'] >= avg_r) & (brand_stats['avg_price'] >= avg_p), 'quadrant'] = 'Premium Leader'
        brand_stats.loc[(brand_stats['avg_rating'] >= avg_r) & (brand_stats['avg_price'] < avg_p), 'quadrant'] = 'Value King'
        brand_stats.loc[(brand_stats['avg_rating'] < avg_r) & (brand_stats['avg_price'] < avg_p), 'quadrant'] = 'Budget/Entry'
        brand_stats.loc[(brand_stats['avg_rating'] < avg_r) & (brand_stats['avg_price'] >= avg_p), 'quadrant'] = 'Overpriced'

    with c4:
        fig3 = px.scatter(
            brand_stats.head(50), x='avg_price', y='avg_rating',
            color='quadrant', hover_name='brand',
            symbol='quadrant',
            title="Brand Strategic Mapping",
            log_x=True,
            color_discrete_map={
                'Premium Leader': '#a78bfa',
                'Value King': '#4ade80',
                'Budget/Entry': '#60a5fa',
                'Overpriced': '#fb7185'
            }
        )
        fig3.add_hline(y=avg_r, line_dash="dot", line_color="#7a7f8e")
        fig3.add_vline(x=avg_p, line_dash="dot", line_color="#7a7f8e")
        fig3.update_layout(**PLOT_THEME)
        st.plotly_chart(fig3, width="stretch")

# PAGE: BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Business Insights":
    st.markdown('<div class="section-header">Market Intelligence Summary</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Value Score = Rating / Price ratio (higher = better value)
        fdf_vs = fdf.copy()
        fdf_vs['value_score'] = (fdf_vs['rating'] / np.log1p(fdf_vs['discounted_price'])).round(3)
        cat_value = fdf_vs.groupby('main_category')['value_score'].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(cat_value, x='value_score', y='main_category',
                     orientation='h', color='value_score',
                     color_continuous_scale=['#1e2129', '#4ade80'],
                     title="Value Score by Category (Rating / log Price)",
                     labels={'value_score': 'Value Score', 'main_category': ''})
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False,
                          title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Correlation matrix
        numeric_cols = ['discounted_price', 'actual_price', 'discount_percentage',
                        'rating', 'rating_count', 'savings']
        corr = fdf[numeric_cols].corr().round(2)
        fig2 = px.imshow(corr, text_auto=True, aspect='auto',
                         color_continuous_scale=['#dc2626', '#0d0f14', '#4ade80'],
                         title="Feature Correlation Matrix",
                         zmin=-1, zmax=1)
        fig2.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'))
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-header">Strategic Recommendations</div>', unsafe_allow_html=True)

    # Compute real insights
    best_value_cat = fdf.copy()
    best_value_cat['value'] = best_value_cat['rating'] / np.log1p(best_value_cat['discounted_price'])
    bv = best_value_cat.groupby('main_category')['value'].mean().idxmax()
    high_disc_low_rating = fdf[(fdf['discount_percentage'] > 50) & (fdf['rating'] < 3.5)]
    high_rev_high_rating = fdf[(fdf['rating_count'] > fdf['rating_count'].quantile(0.8)) & (fdf['rating'] >= 4.3)]

    insights = [
        ("🏆 Best Value Category", f"<b>{bv}</b> offers the best rating-to-price ratio — ideal for budget-conscious buyers", "insight-box"),
        ("⚠️ Discount Trap Products", f"<b>{len(high_disc_low_rating)}</b> products carry 50%+ discounts yet score below 3.5 stars — these are high-risk purchases", "insight-box warn-box"),
        ("🌟 Market Winners", f"<b>{len(high_rev_high_rating)}</b> products combine top-20% review volume with 4.3+ ratings — proven bestsellers worth stocking", "insight-box"),
        ("💡 Price-Rating Paradox", f"Correlation between price and rating is <b>{fdf['discounted_price'].corr(fdf['rating']):.3f}</b> — expensive ≠ better rated on Amazon India", "insight-box"),
        ("📦 Category Depth", f"Electronics leads with <b>{(fdf['main_category']=='Electronics').sum()}</b> products, but Home & Kitchen shows strongest growth potential based on review density", "insight-box"),
        ("🔥 Savings Sweet Spot", f"Products in the ₹500–₹1K range offer average savings of <b>₹{fdf[(fdf['discounted_price'].between(500,1000))]['savings'].mean():,.0f}</b> — highest absolute value for money", "insight-box"),
    ]

    for i in range(0, len(insights), 2):
        c1, c2 = st.columns(2)
        for col, (title, body, cls) in zip([c1, c2], insights[i:i+2]):
            with col:
                st.markdown(f'<div class="{cls}"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Sub-Category Deep Dive</div>', unsafe_allow_html=True)
    subcat_stats = fdf.groupby('sub_category').agg(
        products=('product_id', 'count'),
        avg_price=('discounted_price', 'mean'),
        avg_rating=('rating', 'mean'),
        avg_discount=('discount_percentage', 'mean'),
        total_reviews=('rating_count', 'sum')
    ).reset_index()
    subcat_stats = subcat_stats[subcat_stats['products'] >= 5].sort_values('products', ascending=False).head(20)
    fig3 = px.treemap(subcat_stats, path=['sub_category'], values='products',
                      color='avg_rating',
                      color_continuous_scale=['#dc2626', '#f59e0b', '#4ade80'],
                      title="Sub-Category Treemap (size=products, color=avg rating)",
                      hover_data={'avg_price': ':.0f', 'avg_discount': ':.1f'})
    fig3.update_layout(**PLOT_THEME, title_font=dict(size=14, color='#e8e8e8'), height=450)
    st.plotly_chart(fig3, width="stretch")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.75rem; color:#7a7f8e; padding:0.5rem 0 1.5rem;'>
Amazon India Product Dataset · 1,465 Products · Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
