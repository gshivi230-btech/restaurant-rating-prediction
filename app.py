import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set Page Config
st.set_page_config(
    page_title="Restaurant Rating Prediction | Cognifyz",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .rating-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    .stButton>button {
        background-color: #EF4444;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #DC2626;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Dataset .csv"
MODEL_PATH = PROJECT_ROOT / "model.joblib"
RESULTS_PATH = PROJECT_ROOT / "model_comparison_results.csv"
PREDS_PATH = PROJECT_ROOT / "random_forest_predictions.csv"

@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return df
    return None

@st.cache_resource
def load_model_bundle():
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            return bundle
        except Exception as e:
            st.warning(f"Error loading model bundle: {e}. Rebuilding model...")

    # Fallback: train on the fly if model.joblib is missing or incompatible
    if DATA_PATH.exists():
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        df = pd.read_csv(DATA_PATH)
        df_clean = df.drop(columns=['Rating color', 'Rating text', 'Restaurant ID', 'Restaurant Name', 'Address', 'Locality Verbose'])
        df_clean['Cuisines'] = df_clean['Cuisines'].fillna('Unknown')

        y = df_clean['Aggregate rating']
        X = df_clean.drop(columns=['Aggregate rating'])

        categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        model = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ))
            ]
        )
        model.fit(X_train, y_train)

        bundle = {
            'model': model,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'feature_order': X.columns.tolist(),
            'cities': sorted(df_clean['City'].dropna().unique().tolist()),
            'cuisines': sorted(set([c.strip() for raw in df_clean['Cuisines'].dropna() for c in raw.split(',')])),
            'currencies': sorted(df_clean['Currency'].dropna().unique().tolist()),
            'countries': sorted(df_clean['Country Code'].dropna().unique().tolist())
        }
        try:
            joblib.dump(bundle, MODEL_PATH, compress=3)
        except Exception:
            pass
        return bundle

    return None

df_raw = load_dataset()
bundle = load_model_bundle()

if bundle is None:
    st.error("Model or dataset could not be loaded. Please ensure 'Dataset .csv' or 'model.joblib' is present in the repository.")
    st.stop()

model = bundle['model']
cities = bundle.get('cities', [])
all_cuisines = bundle.get('cuisines', [])
currencies = bundle.get('currencies', [])
countries = bundle.get('countries', [])

# Precompute city lookup for intuitive smart auto-fill
if df_raw is not None:
    city_defaults = df_raw.groupby('City').agg({
        'Country Code': lambda x: x.mode().iloc[0] if not x.empty else 1,
        'Currency': lambda x: x.mode().iloc[0] if not x.empty else 'Indian Rupees(Rs.)',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).to_dict(orient='index')
    all_localities_by_city = df_raw.groupby('City')['Locality'].apply(lambda x: sorted(x.dropna().unique().tolist())).to_dict()
else:
    city_defaults = {}
    all_localities_by_city = {}

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/restaurant.png", width=80)
    st.title("Cognifyz Technologies")
    st.caption("Data Science Internship Project")
    st.markdown("---")
    st.markdown("### 📌 Project Overview")
    st.markdown("""
    **Task:** Restaurant Rating Prediction  
    **Model:** Supervised Random Forest Regressor  
    **Accuracy (R²):** **95.85%**  
    **MAE:** **0.20** | **RMSE:** **0.31**  
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Model Details")
    st.markdown("""
    - **Total Records:** 9,551
    - **Features Evaluated:** 14
    - **Hyperparameters:**
      - Trees: `100`
      - Max Depth: `10`
      - Random State: `42`
    """)
    st.markdown("---")
    st.markdown("🔗 **Repository:** [GitHub](https://github.com/gshivi230-btech/restaurant-rating-prediction)")

# Main Header
st.markdown('<div class="main-header">🍽️ Restaurant Rating Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict customer ratings accurately based on cuisine, price tier, location, and service offerings.</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_predict, tab_insights, tab_explorer = st.tabs([
    "🎯 Predict Rating",
    "📊 Model Performance & Insights",
    "🔍 Dataset Explorer"
])

# ----------------- TAB 1: PREDICT RATING -----------------
with tab_predict:
    st.markdown("### 📝 Enter Restaurant Details")
    st.caption("Fill in the attributes below to generate an estimated aggregate rating (0.0 to 5.0).")

    col_loc, col_biz, col_serv = st.columns([1.2, 1.2, 1])

    with col_loc:
        st.subheader("📍 Location")
        default_city_idx = cities.index("New Delhi") if "New Delhi" in cities else 0
        selected_city = st.selectbox("City", options=cities, index=default_city_idx)

        # Smart defaults based on selected city
        city_info = city_defaults.get(selected_city, {'Country Code': 1, 'Currency': 'Indian Rupees(Rs.)', 'Latitude': 28.6, 'Longitude': 77.2})
        available_localities = all_localities_by_city.get(selected_city, ["Central", "Downtown", "Market Area"])
        
        selected_locality = st.selectbox("Locality / Neighborhood", options=available_localities)
        
        c_lat, c_lon = st.columns(2)
        with c_lat:
            latitude = st.number_input("Latitude", value=float(city_info.get('Latitude', 28.6139)), format="%.4f")
        with c_lon:
            longitude = st.number_input("Longitude", value=float(city_info.get('Longitude', 77.2090)), format="%.4f")
            
        country_code = st.number_input("Country Code", value=int(city_info.get('Country Code', 1)), step=1)

    with col_biz:
        st.subheader("💳 Pricing & Cuisines")
        default_cuisines = ["North Indian", "Chinese"] if ("North Indian" in all_cuisines and "Chinese" in all_cuisines) else [all_cuisines[0]]
        selected_cuisines = st.multiselect(
            "Cuisines Offered",
            options=all_cuisines,
            default=default_cuisines,
            help="Select one or multiple cuisines served by the restaurant."
        )
        cuisines_str = ", ".join(selected_cuisines) if selected_cuisines else "Unknown"

        avg_cost = st.number_input("Average Cost for Two", min_value=0, max_value=500000, value=800, step=50)
        
        curr_options = currencies if currencies else ["Indian Rupees(Rs.)", "Dollar($)"]
        default_curr_idx = curr_options.index(city_info.get('Currency', curr_options[0])) if city_info.get('Currency') in curr_options else 0
        currency = st.selectbox("Currency", options=curr_options, index=default_curr_idx)

        price_range = st.slider("Price Tier (1: Budget ➔ 4: Luxury)", min_value=1, max_value=4, value=2)

    with col_serv:
        st.subheader("🛎️ Services & Votes")
        table_booking = st.radio("Table Booking Available?", ["No", "Yes"], horizontal=True)
        online_delivery = st.radio("Online Delivery Available?", ["No", "Yes"], horizontal=True)
        delivering_now = st.radio("Is Delivering Now?", ["No", "Yes"], horizontal=True)
        switch_menu = "No"

        votes = st.slider("Number of Customer Votes/Reviews", min_value=0, max_value=12000, value=350, step=10,
                         help="Total count of customer reviews previously received.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Calculate Predicted Rating", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            'Country Code': country_code,
            'City': selected_city,
            'Locality': selected_locality,
            'Longitude': longitude,
            'Latitude': latitude,
            'Cuisines': cuisines_str,
            'Average Cost for two': avg_cost,
            'Currency': currency,
            'Has Table booking': table_booking,
            'Has Online delivery': online_delivery,
            'Is delivering now': delivering_now,
            'Switch to order menu': switch_menu,
            'Price range': price_range,
            'Votes': votes
        }])

        try:
            pred_rating = float(model.predict(input_data)[0])
            pred_rating = max(0.0, min(5.0, pred_rating))

            st.markdown("---")
            st.markdown("### 🏆 Prediction Result")

            col_res1, col_res2, col_res3 = st.columns([1, 1.2, 1.5])

            with col_res1:
                st.metric(label="Predicted Aggregate Rating", value=f"{pred_rating:.2f} / 5.0")
                
                # Rating bands matching Cognifyz/Zomato classification
                if pred_rating >= 4.5:
                    badge_color = "#15803D"
                    label = "Dark Green — Excellent ⭐⭐⭐⭐⭐"
                elif pred_rating >= 4.0:
                    badge_color = "#16A34A"
                    label = "Green — Very Good ⭐⭐⭐⭐"
                elif pred_rating >= 3.5:
                    badge_color = "#EAB308"
                    label = "Yellow — Good ⭐⭐⭐"
                elif pred_rating >= 2.5:
                    badge_color = "#F97316"
                    label = "Orange — Average ⭐⭐"
                elif pred_rating >= 1.8:
                    badge_color = "#EF4444"
                    label = "Red — Poor ⭐"
                else:
                    badge_color = "#64748B"
                    label = "White / Grey — Not Rated / Very Low"

                st.markdown(
                    f'<div style="background-color: {badge_color}; color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; text-align: center;">{label}</div>',
                    unsafe_allow_html=True
                )

            with col_res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_rating,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Rating Scale", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': badge_color},
                        'steps': [
                            {'range': [0, 2.5], 'color': "#FEE2E2"},
                            {'range': [2.5, 3.5], 'color': "#FFEDD5"},
                            {'range': [3.5, 4.0], 'color': "#FEF08A"},
                            {'range': [4.0, 4.5], 'color': "#DCFCE7"},
                            {'range': [4.5, 5.0], 'color': "#BBF7D0"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': pred_rating
                        }
                    }
                ))
                fig_gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_res3:
                st.markdown("#### 💡 Prediction Insights")
                st.markdown(f"""
                - **Model Confidence Range:** `{max(0.0, pred_rating - 0.31):.2f}` – `{min(5.0, pred_rating + 0.31):.2f}` (RMSE = ±0.31)
                - **Primary Influence:** Customer review count (`{votes}` votes) is the single most decisive factor in determining high aggregate scores.
                - **Service Impact:** Table booking (`{table_booking}`) and Online delivery (`{online_delivery}`) provide notable boosts for mid-to-high tier venues.
                - **Price-to-Cost Alignment:** Price Tier `{price_range}` with Average Cost for two `{avg_cost} {currency}`.
                """)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------- TAB 2: MODEL PERFORMANCE -----------------
with tab_insights:
    st.markdown("### 📊 Supervised Regression Model Evaluation")
    st.write("Comparison of baseline Linear Regression, Decision Tree Regressor, and the chosen Random Forest Regressor on the 20% test holdout split.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Champion Model", "Random Forest")
    with col_m2:
        st.metric("R² Score (Variance Explained)", "95.85%", delta="+48.56% vs Linear")
    with col_m3:
        st.metric("Mean Absolute Error (MAE)", "0.2006", delta="-0.6895", delta_color="inverse")
    with col_m4:
        st.metric("Root Mean Squared Error (RMSE)", "0.3075", delta="-0.7878", delta_color="inverse")

    st.markdown("---")

    # Load comparison table
    if RESULTS_PATH.exists():
        comp_df = pd.read_csv(RESULTS_PATH)
        # Rename column if corrupted encoding
        comp_df.columns = [c if "R" not in c or "2" in c else "R2 Score" for c in comp_df.columns]
    else:
        comp_df = pd.DataFrame({
            "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
            "MAE": [0.8901, 0.2176, 0.2006],
            "MSE": [1.1998, 0.1100, 0.0946],
            "RMSE": [1.0954, 0.3317, 0.3075],
            "R2 Score": [0.4729, 0.9517, 0.9585]
        })

    col_c1, col_c2 = st.columns([1, 1.2])

    with col_c1:
        st.markdown("#### 📋 Model Performance Table")
        st.dataframe(comp_df.style.highlight_min(subset=["MAE", "MSE", "RMSE"], color="#DCFCE7").highlight_max(subset=["R2 Score"], color="#DCFCE7"), use_container_width=True)

        fig_r2 = px.bar(
            comp_df,
            x="Model",
            y="R2 Score",
            color="Model",
            text="R2 Score",
            title="Model Accuracy Comparison (R² Score)",
            color_discrete_sequence=["#94A3B8", "#60A5FA", "#16A34A"]
        )
        fig_r2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_r2.update_layout(yaxis_range=[0, 1.1], showlegend=False, height=300)
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_c2:
        st.markdown("#### 📉 Error Metrics (Lower is Better)")
        fig_err = px.bar(
            comp_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error"),
            x="Model",
            y="Error",
            color="Metric",
            barmode="group",
            title="MAE & RMSE Error Comparison",
            color_discrete_sequence=["#F59E0B", "#EF4444"]
        )
        fig_err.update_layout(height=400)
        st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("---")
    
    # Feature Importance & Actual vs Predicted
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("#### 🌟 Top Feature Importances")
        try:
            forest = model.named_steps['model']
            preprocessor = model.named_steps['preprocessor']
            feat_names = preprocessor.get_feature_names_out()
            importances = forest.feature_importances_
            feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(12)
            
            # Clean up feature names for clean presentation
            feat_df["Clean Name"] = feat_df["Feature"].apply(lambda x: x.replace("num__", "").replace("cat__", ""))
            
            fig_imp = px.bar(
                feat_df.sort_values(by="Importance", ascending=True),
                x="Importance",
                y="Clean Name",
                orientation='h',
                title="Top Influential Features in Prediction",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            fig_imp.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.info("Feature importance breakdown available when model is loaded.")

    with col_f2:
        st.markdown("#### 🎯 Actual vs Predicted Ratings (Test Set)")
        if PREDS_PATH.exists():
            preds_df = pd.read_csv(PREDS_PATH).sample(n=min(500, len(pd.read_csv(PREDS_PATH))), random_state=42)
            fig_scatter = px.scatter(
                preds_df,
                x="Actual Rating",
                y="Predicted Rating",
                opacity=0.6,
                title="Actual vs Predicted Scatter Plot (Test Holdout)",
                color_discrete_sequence=["#2563EB"]
            )
            fig_scatter.add_trace(go.Scatter(
                x=[0, 5],
                y=[0, 5],
                mode='lines',
                name='Perfect Fit (y = x)',
                line=dict(color='red', dash='dash')
            ))
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Actual vs Predicted scatter plot generated upon batch evaluation.")

# ----------------- TAB 3: DATASET EXPLORER -----------------
with tab_explorer:
    st.markdown("### 🔍 Dataset Explorer & Live Samples")
    if df_raw is not None:
        col_flt1, col_flt2, col_flt3 = st.columns(3)
        with col_flt1:
            filter_city = st.selectbox("Filter by City", options=["All"] + cities)
        with col_flt2:
            rating_range = st.slider("Filter by Aggregate Rating", min_value=0.0, max_value=5.0, value=(0.0, 5.0), step=0.1)
        with col_flt3:
            price_filter = st.multiselect("Price Range Filter", options=[1, 2, 3, 4], default=[1, 2, 3, 4])

        filtered_df = df_raw.copy()
        if filter_city != "All":
            filtered_df = filtered_df[filtered_df["City"] == filter_city]
        filtered_df = filtered_df[
            (filtered_df["Aggregate rating"] >= rating_range[0]) &
            (filtered_df["Aggregate rating"] <= rating_range[1]) &
            (filtered_df["Price range"].isin(price_filter))
        ]

        st.caption(f"Showing {len(filtered_df):,} matching restaurants out of {len(df_raw):,} total.")
        
        display_cols = ["Restaurant Name", "City", "Locality", "Cuisines", "Price range", "Average Cost for two", "Votes", "Aggregate rating"]
        available_display_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[available_display_cols].head(100), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎲 Test Random Record from Dataset")
        if st.button("Pick a Random Restaurant and Test Prediction"):
            random_sample = df_raw.sample(n=1, random_state=None).iloc[0]
            st.write(f"**Restaurant:** {random_sample.get('Restaurant Name', 'Unknown')} ({random_sample.get('City', '')})")
            
            clean_sample_df = pd.DataFrame([{
                'Country Code': random_sample['Country Code'],
                'City': random_sample['City'],
                'Locality': random_sample['Locality'],
                'Longitude': random_sample['Longitude'],
                'Latitude': random_sample['Latitude'],
                'Cuisines': random_sample.get('Cuisines', 'Unknown'),
                'Average Cost for two': random_sample['Average Cost for two'],
                'Currency': random_sample['Currency'],
                'Has Table booking': random_sample['Has Table booking'],
                'Has Online delivery': random_sample['Has Online delivery'],
                'Is delivering now': random_sample['Is delivering now'],
                'Switch to order menu': random_sample['Switch to order menu'],
                'Price range': random_sample['Price range'],
                'Votes': random_sample['Votes']
            }])
            sample_pred = float(model.predict(clean_sample_df)[0])
            actual_rating = float(random_sample['Aggregate rating'])
            
            c_rand1, c_rand2, c_rand3 = st.columns(3)
            with c_rand1:
                st.metric("Actual Rating", f"{actual_rating:.2f}")
            with c_rand2:
                st.metric("Predicted Rating", f"{sample_pred:.2f}")
            with c_rand3:
                error = abs(actual_rating - sample_pred)
                st.metric("Absolute Difference", f"{error:.2f}", delta="-Good" if error <= 0.35 else "Acceptable")
    else:
        st.info("Dataset .csv is not available.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.9rem; padding: 10px;">
    Cognifyz Technologies Data Science Internship | Machine Learning Track<br>
    Built with Python, Streamlit, Scikit-learn, and Plotly
</div>
""", unsafe_allow_html=True)
