import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
from io import BytesIO
import time

# Set page config
st.set_page_config(
    page_title="AI-Powered Sales Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .st-bq {
        border-radius: 10px;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        background-image: none;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .header-text {
        color: #1e3a8a;
    }
    .insight-card {
        background: #f0f7ff;
        border-left: 4px solid #1e88e5;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
        color: #333333 !important;
    }
    .insight-card * {
        color: #333333 !important;
    }
    .sidebar .sidebar-content {
        overflow-y: auto;
        max-height: 100vh;
    }
    @media (max-width: 768px) {
        .sidebar .sidebar-content {
            max-height: none;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for KAM selection and uploaded data
if 'selected_kam' not in st.session_state:
    st.session_state.selected_kam = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Sidebar - KAM Login Section
with st.sidebar:
    st.header("Key Account Manager Login")
    
    # File uploader for KAMs to upload their data (both CSV and Excel)
    uploaded_file = st.file_uploader("Upload your sales data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.uploaded_data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                st.session_state.uploaded_data = pd.read_excel(uploaded_file)
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # If no file uploaded, use default data
    if st.session_state.uploaded_data is None:
        # Load default data function with caching and error handling
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def load_data():
            try:
                # Read the CSV file
                df = pd.read_csv('Forecast.csv')
                
                # Data cleaning and preprocessing
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                df = df.dropna(subset=['Date'])  # Drop rows with invalid dates
                
                # Create time-based features
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Quarter'] = df['Date'].dt.quarter
                
                # Convert currency columns to numeric
                currency_cols = ['Confirmed Value', 'Budget Value', 'USD Net Selling Price', 'USD Sales Value', 'USD Outlook Value']
                for col in currency_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Convert quantity columns to numeric
                quantity_cols = ['Confirmed Quantity', 'Budget Quantity', 'Order Quantity', 'Outlook Quantity']
                for col in quantity_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill missing values
                num_cols = ['Confirmed Quantity', 'Confirmed Value', 'Budget Quantity', 'Budget Value']
                for col in num_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                
                return df
            
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return pd.DataFrame()  # Return empty dataframe on error

        # Load default data
        df = load_data()
    else:
        df = st.session_state.uploaded_data.copy()
        # Perform similar data cleaning on uploaded data
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            
            # Convert numeric columns
            numeric_cols = ['Confirmed Quantity', 'Confirmed Value', 'Budget Quantity', 'Budget Value', 
                          'USD Net Selling Price', 'USD Sales Value', 'USD Outlook Value', 'Order Quantity', 
                          'Outlook Quantity']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        except Exception as e:
            st.error(f"Error processing uploaded data: {str(e)}")

    # KAM selection dropdown
    kam_list = ['Select KAM'] + sorted(df['Key Account Manager'].dropna().unique().tolist())
    selected_kam = st.selectbox("Select Your Name", kam_list, index=0)

    if selected_kam != 'Select KAM':
        st.session_state.selected_kam = selected_kam
        st.success(f"Logged in as: {st.session_state.selected_kam}")

    # Only show filters if KAM is selected
    if st.session_state.selected_kam:
        st.header("Advanced Filters")
        
        # Collapsible filter sections
        with st.expander("Date Range", expanded=True):
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            date_range = st.date_input(
                "Select Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
        
        with st.expander("Organizational Filters"):
            dos_options = ['All'] + sorted(df['DOS'].dropna().unique().tolist())
            selected_dos = st.selectbox("DOS", dos_options, index=0)
            
            vp_options = ['All'] + sorted(df['Vice President'].dropna().unique().tolist())
            selected_vp = st.selectbox("Vice President", vp_options, index=0)
            
            sales_org_options = ['All'] + sorted(df['Sales Organization'].dropna().unique().tolist())
            selected_sales_org = st.selectbox("Sales Organization", sales_org_options, index=0)
        
        with st.expander("Product Filters"):
            mg1_options = ['All'] + sorted(df['Material Group1 MST'].dropna().unique().tolist())
            selected_mg1 = st.selectbox("Material Group 1", mg1_options, index=0)
            
            mg4_options = ['All'] + sorted(df['Material Group4 MST'].dropna().unique().tolist())
            selected_mg4 = st.selectbox("Material Group 4", mg4_options, index=0)
            
            product_type_options = ['All'] + sorted(df['Product Type'].dropna().unique().tolist())
            selected_product_type = st.selectbox("Product Type", product_type_options, index=0)
        
        with st.expander("Customer Filters"):
            customer_options = ['All'] + sorted(df['Customer'].dropna().unique().tolist())
            selected_customer = st.selectbox("Customer", customer_options, index=0)
            
            search_term_options = ['All'] + sorted(df['Search Term 2'].dropna().unique().tolist())
            selected_search_term = st.selectbox("Search Term", search_term_options, index=0)

# Main page header
st.title("🧠 AI-Powered Sales Intelligence Dashboard")

if st.session_state.selected_kam:
    # Filter data for the selected KAM
    kam_df = df[df['Key Account Manager'] == st.session_state.selected_kam]
    
    if kam_df.empty:
        st.warning("No data available for the selected KAM")
        st.stop()
    
    # Apply filters
    filtered_df = kam_df.copy()
    
    # Date filter
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
            (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Other filters
    filter_mapping = {
        'DOS': selected_dos,
        'Vice President': selected_vp,
        'Sales Organization': selected_sales_org,
        'Material Group1 MST': selected_mg1,
        'Material Group4 MST': selected_mg4,
        'Product Type': selected_product_type,
        'Customer': selected_customer,
        'Search Term 2': selected_search_term
    }
    
    for col, value in filter_mapping.items():
        if value != 'All':
            filtered_df = filtered_df[filtered_df[col] == value]
    
    if filtered_df.empty:
        st.warning("No data matches your filters. Please adjust your selection.")
        st.stop()
    
    # Metrics row with error handling
    st.markdown("### KAM Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            total_sales = filtered_df['Confirmed Value'].sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Sales", f"${total_sales:,.0f}" if total_sales == total_sales else "N/A")  # Check for NaN
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Sales", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        try:
            total_qty = filtered_df['Confirmed Quantity'].sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Quantity", f"{total_qty:,.0f}" if total_qty == total_qty else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Quantity", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        try:
            avg_price = filtered_df['USD Net Selling Price'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. Selling Price", f"${avg_price:,.2f}" if avg_price == avg_price else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. Selling Price", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        try:
            num_customers = filtered_df['Customer'].nunique()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Customers", num_customers)
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Customers", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Insights Section with loading spinner
    st.markdown("## 🧠 AI-Powered Insights")
    
    def generate_insights(data):
        insights = []
        
        try:
            # 1. Top performing products
            if 'Material Group4 MST' in data.columns and 'Confirmed Value' in data.columns:
                top_products = data.groupby('Material Group4 MST')['Confirmed Value'].sum().nlargest(3)
                if len(top_products) > 0:
                    total_sales = data['Confirmed Value'].sum()
                    if total_sales > 0:
                        insight = f"**Top Performing Products**: {', '.join(top_products.index.tolist())} account for ${top_products.sum():,.0f} in sales ({top_products.sum()/total_sales:.0%} of total)."
                        insights.append(insight)
        
            # 2. Customer concentration
            if 'Customer' in data.columns and 'Confirmed Value' in data.columns:
                top_customers = data.groupby('Customer')['Confirmed Value'].sum().nlargest(3)
                if len(top_customers) > 0:
                    insight = f"**Key Customers**: {top_customers.index[0]} is your top customer with ${top_customers.iloc[0]:,.0f} in sales."
                    if len(top_customers) > 2:
                        insight += f" Consider expanding relationships with {top_customers.index[1]} and {top_customers.index[2]}."
                    insights.append(insight)
        
            # 3. Budget vs Actual
            if 'Budget Value' in data.columns and 'Confirmed Value' in data.columns:
                budget_total = data['Budget Value'].sum()
                if budget_total > 0:
                    variance = (data['Confirmed Value'].sum() - budget_total) / budget_total
                    if variance > 0:
                        insight = f"**Performance**: You're exceeding budget by {variance:.0%}. Great job!"
                    else:
                        insight = f"**Performance**: Currently {abs(variance):.0%} below budget. Focus on high-margin products to close the gap."
                    insights.append(insight)
        
            # 4. Seasonality detection
            if 'Date' in data.columns and 'Confirmed Value' in data.columns:
                monthly_sales = data.groupby(data['Date'].dt.to_period('M'))['Confirmed Value'].sum()
                if len(monthly_sales) > 6:
                    peak_month = monthly_sales.idxmax().strftime('%B')
                    insight = f"**Seasonality**: Your sales peak in {peak_month}. Plan inventory and promotions accordingly."
                    insights.append(insight)
        
            # 5. Product mix analysis
            if 'Material Group4 MST' in data.columns and 'Confirmed Value' in data.columns:
                product_mix = data.groupby('Material Group4 MST')['Confirmed Value'].sum().sort_values(ascending=False)
                if len(product_mix) > 3:
                    insight = f"**Product Mix**: Consider diversifying beyond {product_mix.index[0]} which dominates your portfolio."
                    insights.append(insight)
        
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            insights.append("**Error**: Some insights could not be generated due to data issues.")
        
        return insights if insights else ["No significant insights could be generated from the current data."]
    
    # Display insights with error handling
    try:
        with st.spinner("Generating insights..."):
            insights = generate_insights(filtered_df)
            for insight in insights:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
    except:
        st.error("Failed to generate insights")

    # AI Suggestions Button with proper error handling
    if st.button("🧠 Generate Custom AI Suggestions"):
        try:
            with st.spinner("Analyzing your data with AI..."):
                # Simulate AI processing time
                time.sleep(2)
                
                # Prepare data summary for suggestions
                data_summary = {
                    "total_sales": filtered_df['Confirmed Value'].sum(),
                    "top_customers": filtered_df.groupby('Customer')['Confirmed Value'].sum().nlargest(3).to_dict(),
                    "top_products": filtered_df.groupby('Material Group4 MST')['Confirmed Value'].sum().nlargest(3).to_dict(),
                    "sales_trend": filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Confirmed Value'].sum().to_dict(),
                }
                
                # Generate suggestions (simulated)
                suggestions = [
                    "Based on your top performing products, consider bundling them with complementary items to increase average order value.",
                    f"Your customer {list(data_summary['top_customers'].keys())[0] if data_summary['top_customers'] else 'a key customer'} has significant potential for upselling - schedule a business review meeting.",
                    "The data shows untapped potential in the " + (filtered_df['DOS'].mode()[0] if 'DOS' in filtered_df.columns else 'certain') + " region - consider targeted marketing campaigns.",
                    "Implement a customer loyalty program for your top customers to improve retention.",
                    "Analyze why " + (filtered_df[filtered_df['Confirmed Value'] == 0]['Material Group4 MST'].mode()[0] if 'Material Group4 MST' in filtered_df.columns else 'certain') + " products aren't selling and adjust your strategy."
                ]
                
                st.markdown("### 🤖 AI Recommendations")
                for suggestion in suggestions[:3]:  # Show top 3 suggestions
                    st.markdown(f'<div class="insight-card">✨ {suggestion}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to generate AI suggestions: {str(e)}")

    # Tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["📊 Sales Performance", "📈 Forecasting", "🔍 Deep Dive"])
    
    with tab1:
        st.markdown("### Sales Performance Analysis")
        
        # Time aggregation selection
        time_agg = st.radio(
            "Time Aggregation",
            ["Daily", "Weekly", "Monthly", "Quarterly"],
            horizontal=True,
            index=2
        )
        
        try:
            # Prepare data based on time aggregation
            if time_agg == "Daily":
                agg_df = filtered_df.groupby('Date').agg({
                    'Confirmed Quantity': 'sum',
                    'Confirmed Value': 'sum'
                }).reset_index()
                x_axis = 'Date'
            elif time_agg == "Weekly":
                filtered_df['Week'] = filtered_df['Date'].dt.to_period('W').dt.start_time
                agg_df = filtered_df.groupby('Week').agg({
                    'Confirmed Quantity': 'sum',
                    'Confirmed Value': 'sum'
                }).reset_index()
                x_axis = 'Week'
            elif time_agg == "Monthly":
                filtered_df['Month'] = filtered_df['Date'].dt.to_period('M').dt.start_time
                agg_df = filtered_df.groupby('Month').agg({
                    'Confirmed Quantity': 'sum',
                    'Confirmed Value': 'sum'
                }).reset_index()
                x_axis = 'Month'
            else:  # Quarterly
                filtered_df['Quarter'] = filtered_df['Date'].dt.to_period('Q').dt.start_time
                agg_df = filtered_df.groupby('Quarter').agg({
                    'Confirmed Quantity': 'sum',
                    'Confirmed Value': 'sum'
                }).reset_index()
                x_axis = 'Quarter'
            
            # Chart selection
            chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Area Chart"])
            
            # Create chart with error handling
            try:
                if not agg_df.empty:
                    if chart_type == "Line Chart":
                        fig = px.line(agg_df, x=x_axis, y='Confirmed Value', 
                                     title=f'Sales Value Trend ({time_agg})',
                                     labels={'Confirmed Value': 'Sales Value ($)'})
                    elif chart_type == "Bar Chart":
                        fig = px.bar(agg_df, x=x_axis, y='Confirmed Value', 
                                    title=f'Sales Value Trend ({time_agg})',
                                    labels={'Confirmed Value': 'Sales Value ($)'})
                    else:  # Area Chart
                        fig = px.area(agg_df, x=x_axis, y='Confirmed Value', 
                                     title=f'Sales Value Trend ({time_agg})',
                                     labels={'Confirmed Value': 'Sales Value ($)'})
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for the selected time aggregation")
            except Exception as e:
                st.error(f"Failed to create chart: {str(e)}")
            
            # Performance by different dimensions
            st.markdown("### Performance by Dimension")
            dimension = st.selectbox(
                "Analyze by",
                ["Customer", "Product Category", "Sales Organization", "Material Group"],
                index=0
            )
            
            if dimension == "Material Group":
                dimension_col = 'Material Group4 MST'
            elif dimension == "Product Category":
                dimension_col = 'Product Type'
            else:
                dimension_col = dimension
            
            try:
                if dimension_col in filtered_df.columns:
                    performance_df = filtered_df.groupby(dimension_col).agg({
                        'Confirmed Value': 'sum',
                        'Confirmed Quantity': 'sum',
                        'USD Net Selling Price': 'mean'
                    }).sort_values('Confirmed Value', ascending=False).reset_index()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### Top {dimension}s by Sales")
                        if not performance_df.empty:
                            fig_dim1 = px.bar(performance_df.head(10), 
                                             x=dimension_col, 
                                             y='Confirmed Value',
                                             color='Confirmed Value',
                                             color_continuous_scale='Blues')
                            st.plotly_chart(fig_dim1, use_container_width=True)
                        else:
                            st.warning("No data available for this dimension")
                    
                    with col2:
                        st.markdown(f"#### {dimension} Price Distribution")
                        if not filtered_df.empty and 'USD Net Selling Price' in filtered_df.columns:
                            fig_dim2 = px.box(filtered_df, 
                                             x=dimension_col, 
                                             y='USD Net Selling Price',
                                             points=False)
                            st.plotly_chart(fig_dim2, use_container_width=True)
                        else:
                            st.warning("No price data available for this dimension")
                else:
                    st.warning(f"Column '{dimension_col}' not found in data")
            except Exception as e:
                st.error(f"Error analyzing by dimension: {str(e)}")
        
        except Exception as e:
            st.error(f"Error in sales performance analysis: {str(e)}")
    
    with tab2:
        st.markdown("## AI Sales Forecasting")
        
        try:
            # Model selection
            model_type = st.radio(
                "Select Forecasting Model",
                ["Prophet (Time Series)", "Random Forest"],
                horizontal=True
            )
            
            # Forecast parameters
            st.markdown("### Forecast Parameters")
            col1, col2 = st.columns(2)
            with col1:
                # Enhanced forecast horizon selection with time period options
                forecast_period = st.selectbox(
                    "Forecast Time Period",
                    ["Days", "Weeks", "Months", "Quarters"],
                    index=2
                )
                
                forecast_horizon = st.slider(
                    f"Number of {forecast_period} to Forecast",
                    min_value=1,
                    max_value=24,
                    value=6,
                    help=f"Number of future {forecast_period.lower()} to forecast"
                )
                
            with col2:
                if model_type == "Prophet (Time Series)":
                    seasonality_mode = st.selectbox(
                        "Seasonality Mode",
                        ["additive", "multiplicative"],
                        help="How seasonality components are modeled"
                    )
                else:
                    test_size = st.slider(
                        "Test Set Size (%)",
                        min_value=10,
                        max_value=50,
                        value=20,
                        help="Percentage of data to use for testing model accuracy"
                    )
            
            # Determine frequency based on selected period
            freq_map = {
                "Days": "D",
                "Weeks": "W",
                "Months": "M",
                "Quarters": "Q"
            }
            forecast_freq = freq_map[forecast_period]
            
            if model_type == "Prophet (Time Series)":
                try:
                    # Prepare data for Prophet
                    prophet_df = filtered_df.groupby('Date')['Confirmed Value'].sum().reset_index()
                    prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Confirmed Value': 'y'})
                    
                    if len(prophet_df) < 2:
                        st.warning("Insufficient data for forecasting. Need at least 2 data points.")
                    else:
                        with st.spinner("Training Prophet model..."):
                            model = Prophet(seasonality_mode=seasonality_mode)
                            model.fit(prophet_df)
                            
                            # Create future dataframe with selected frequency
                            future = model.make_future_dataframe(
                                periods=forecast_horizon,
                                freq=forecast_freq
                            )
                            
                            # Make forecast
                            forecast = model.predict(future)
                            
                            # Show forecast plot
                            st.markdown("### Sales Forecast")
                            fig_forecast = model.plot(forecast)
                            st.pyplot(fig_forecast)
                            
                            # Show forecast components
                            st.markdown("### Forecast Components")
                            fig_components = model.plot_components(forecast)
                            st.pyplot(fig_components)
                            
                            # Show forecast values
                            st.markdown("### Forecast Values")
                            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon))
                except Exception as e:
                    st.error(f"Prophet forecasting failed: {str(e)}")
            
            else:  # Random Forest
                try:
                    # Prepare features for Random Forest
                    forecast_df = filtered_df.copy()
                    forecast_df = forecast_df.groupby('Date')['Confirmed Value'].sum().reset_index()
                    forecast_df['Day'] = forecast_df['Date'].dt.day
                    forecast_df['Month'] = forecast_df['Date'].dt.month
                    forecast_df['Year'] = forecast_df['Date'].dt.year
                    forecast_df['DayOfWeek'] = forecast_df['Date'].dt.dayofweek
                    forecast_df['DayOfYear'] = forecast_df['Date'].dt.dayofyear
                    
                    if len(forecast_df) < 10:
                        st.warning("Insufficient data for Random Forest. Need at least 10 data points.")
                    else:
                        # Split data
                        X = forecast_df[['Day', 'Month', 'Year', 'DayOfWeek', 'DayOfYear']]
                        y = forecast_df['Confirmed Value']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42
                        )
                        
                        with st.spinner("Training Random Forest model..."):
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            # Show results
                            st.markdown("### Model Performance")
                            st.metric("Mean Absolute Error", f"${mae:,.2f}" if mae == mae else "N/A")
                            
                            # Feature importance
                            st.markdown("### Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': X.columns,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig_importance = px.bar(importance_df, 
                                                x='Importance', 
                                                y='Feature',
                                                orientation='h',
                                                title='Feature Importance for Forecasting')
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Create future dates for forecasting with selected frequency
                            last_date = forecast_df['Date'].max()
                            if forecast_period == "Days":
                                future_dates = pd.date_range(
                                    start=last_date + pd.Timedelta(days=1),
                                    periods=forecast_horizon,
                                    freq='D'
                                )
                            elif forecast_period == "Weeks":
                                future_dates = pd.date_range(
                                    start=last_date + pd.Timedelta(weeks=1),
                                    periods=forecast_horizon,
                                    freq='W-MON'
                                )
                            elif forecast_period == "Months":
                                future_dates = pd.date_range(
                                    start=last_date + pd.DateOffset(months=1),
                                    periods=forecast_horizon,
                                    freq='MS'
                                )
                            else:  # Quarters
                                future_dates = pd.date_range(
                                    start=last_date + pd.DateOffset(months=3),
                                    periods=forecast_horizon,
                                    freq='QS'
                                )
                            
                            # Prepare future features
                            future_df = pd.DataFrame({
                                'Date': future_dates,
                                'Day': future_dates.day,
                                'Month': future_dates.month,
                                'Year': future_dates.year,
                                'DayOfWeek': future_dates.dayofweek,
                                'DayOfYear': future_dates.dayofyear
                            })
                            
                            # Make forecasts
                            future_pred = model.predict(future_df[['Day', 'Month', 'Year', 'DayOfWeek', 'DayOfYear']])
                            future_df['Forecast'] = future_pred
                            
                            # Combine historical and forecast data
                            history_df = forecast_df[['Date', 'Confirmed Value']].rename(columns={'Confirmed Value': 'Actual'})
                            future_df = future_df[['Date', 'Forecast']]
                            combined_df = pd.concat([
                                history_df,
                                future_df.rename(columns={'Forecast': 'Actual'})
                            ], ignore_index=True)
                            
                            # Plot results
                            fig = px.line(combined_df, x='Date', y='Actual', title='Sales Forecast (Random Forest)')
                            fig.add_scatter(
                                x=future_df['Date'], 
                                y=future_df['Forecast'], 
                                mode='lines', 
                                name='Forecast',
                                line=dict(color='red', dash='dash')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show forecast values
                            st.markdown("### Forecast Values")
                            st.dataframe(future_df)
                except Exception as e:
                    st.error(f"Random Forest forecasting failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error in forecasting tab: {str(e)}")
    
    with tab3:
        st.markdown("## Data Deep Dive")
        
        try:
            # Raw data explorer
            st.markdown("### Filtered Data Preview")
            st.dataframe(filtered_df.head(1000))  # Limit preview to 1000 rows
            
            # Data export with proper Excel handling
            st.markdown("### Export Data")
            export_format = st.radio(
                "Select Export Format",
                ["CSV", "Excel"],
                horizontal=True
            )
            
            if st.button("Export Data"):
                if export_format == "CSV":
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="sales_data.csv",
                        mime="text/csv"
                    )
                else:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name='SalesData')
                    excel_data = output.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name="sales_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Advanced analysis
            st.markdown("### Advanced Analysis")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Customer Segmentation", "Product Performance", "Sales Trend Anomalies"],
                index=0
            )
            
            if analysis_type == "Customer Segmentation":
                try:
                    st.markdown("#### Customer Segmentation by Value")
                    if 'Customer' in filtered_df.columns and 'Confirmed Value' in filtered_df.columns:
                        customer_value = filtered_df.groupby('Customer')['Confirmed Value'].sum().sort_values(ascending=False)
                        if not customer_value.empty:
                            fig_cust = px.treemap(
                                customer_value.reset_index(),
                                path=['Customer'],
                                values='Confirmed Value',
                                title='Customer Value Distribution'
                            )
                            st.plotly_chart(fig_cust, use_container_width=True)
                        else:
                            st.warning("No customer data available")
                    else:
                        st.warning("Required columns not found in data")
                except Exception as e:
                    st.error(f"Customer segmentation failed: {str(e)}")
            
            elif analysis_type == "Product Performance":
                try:
                    st.markdown("#### Product Performance Matrix")
                    if ('Material Group4 MST' in filtered_df.columns and 
                        'Confirmed Value' in filtered_df.columns and 
                        'Confirmed Quantity' in filtered_df.columns):
                        product_perf = filtered_df.groupby('Material Group4 MST').agg({
                            'Confirmed Value': 'sum',
                            'Confirmed Quantity': 'sum'
                        }).reset_index()
                        product_perf['Avg Price'] = product_perf['Confirmed Value'] / product_perf['Confirmed Quantity']
                        
                        if not product_perf.empty:
                            fig_prod = px.scatter(
                                product_perf,
                                x='Confirmed Quantity',
                                y='Avg Price',
                                size='Confirmed Value',
                                color='Material Group4 MST',
                                hover_name='Material Group4 MST',
                                title='Product Performance Matrix (Quantity vs Price)'
                            )
                            st.plotly_chart(fig_prod, use_container_width=True)
                        else:
                            st.warning("No product performance data available")
                    else:
                        st.warning("Required columns not found in data")
                except Exception as e:
                    st.error(f"Product performance analysis failed: {str(e)}")
            
            else:  # Sales Trend Anomalies
                try:
                    st.markdown("#### Sales Trend Anomalies Detection")
                    if 'Date' in filtered_df.columns and 'Confirmed Value' in filtered_df.columns:
                        ts_df = filtered_df.groupby('Date')['Confirmed Value'].sum().reset_index()
                        if len(ts_df) >= 7:  # Need at least 7 points for rolling stats
                            ts_df['RollingAvg'] = ts_df['Confirmed Value'].rolling(window=7).mean()
                            ts_df['StdDev'] = ts_df['Confirmed Value'].rolling(window=7).std()
                            ts_df['UpperBound'] = ts_df['RollingAvg'] + (ts_df['StdDev'] * 2)
                            ts_df['LowerBound'] = ts_df['RollingAvg'] - (ts_df['StdDev'] * 2)
                            ts_df['Anomaly'] = (ts_df['Confirmed Value'] > ts_df['UpperBound']) | (ts_df['Confirmed Value'] < ts_df['LowerBound'])
                            
                            fig_anomaly = px.line(ts_df, x='Date', y=['Confirmed Value', 'RollingAvg', 'UpperBound', 'LowerBound'])
                            fig_anomaly.add_scatter(
                                x=ts_df[ts_df['Anomaly']]['Date'],
                                y=ts_df[ts_df['Anomaly']]['Confirmed Value'],
                                mode='markers',
                                marker=dict(color='red', size=10),
                                name='Anomaly'
                            )
                            st.plotly_chart(fig_anomaly, use_container_width=True)
                        else:
                            st.warning("Need at least 7 days of data for anomaly detection")
                    else:
                        st.warning("Required columns not found in data")
                except Exception as e:
                    st.error(f"Anomaly detection failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error in deep dive tab: {str(e)}")

else:
    st.warning("Please select your name from the sidebar to access the dashboard")

# Footer
st.markdown("---")
st.markdown("""
**AI-Powered Sales Intelligence Dashboard**  
*For Key Account Managers* | [Documentation](#) | [Feedback](#)
""")
