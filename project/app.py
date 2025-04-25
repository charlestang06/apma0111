# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.ticker as ticker
import data_processing  # new import to use process_data


def plot_conf_matrix(cm, labels=None, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(data=cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # Adjust tick labels for better readability
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=cm.shape[1]))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=cm.shape[0]))
    st.pyplot(fig)


# Page configuration
st.set_page_config(page_title="Mortgage Approval Analysis", layout="wide")

# Load data with caching

all_features = None  # Initialize all_features to None


# Add file uploader for raw CSV input
st.sidebar.header("Upload Raw CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload your state's raw data CSV", type="csv")

if uploaded_file is not None:
    # Create a hash of the file content as a simple identifier
    current_hash = hash(uploaded_file.getvalue())
    if st.session_state.get("uploaded_file_hash") != current_hash:
        st.session_state["uploaded_file_hash"] = current_hash
        st.session_state["raw_df"] = pd.read_csv(uploaded_file)
        st.session_state["processed_df"] = data_processing.process_data(
            st.session_state["raw_df"])
    st.subheader("Raw Data Preview")
    st.dataframe(st.session_state["raw_df"].head())
    st.success("Data processed successfully!")
    st.subheader("Processed Data Preview")
    st.dataframe(st.session_state["processed_df"].head())
    df = st.session_state["processed_df"]  # use stored  processed data
else:
    # Existing caching mechanism loads processed_data.csv
    @st.cache_data
    def load_data():
        df = pd.read_csv("processed_data.csv", index_col=0)
        df = df[df["derived_race"] != "Free Form Text Only"]
        df = df[df["derived_ethnicity"] != "Free Form Text Only"]
        return df
    df = load_data()

features = ['derived_ethnicity', 'derived_race', 'derived_sex', 'loan_type',
            'loan_purpose', 'lien_status', 'loan_amount', 'loan_to_value_ratio',
            'interest_rate', 'property_value', 'occupancy_type', 'debt_to_income_ratio']
target = 'loan_approved'

X = df[features]
y = df[target].astype('category').cat.codes

# Preprocessing
categorical_cols = X.select_dtypes(
    include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Sidebar for navigation
# Enhance sidebar styling with custom CSS
st.sidebar.markdown(
    """
    <style>
    .stSidebar .sidebar-content {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 8px;
    }
    .stSidebar .stRadio > div {
        margin-bottom: 20px;
    }
    .stSidebar .stRadio > label {
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True
)
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Background",
    "Exploratory Analysis",
    "Logistic Regression Model",
    "XGBoost Model",
])

# Main content
if section == "Background":
    st.title("Background")
    st.markdown("""
    Homeownership is a key component of the American Dream, not only providing stability and pride but also helping families build generational wealth. Structural injustice, however, permeates the housing market, strongly disfavoring marginalized minority groups such as blacks and Hispanics and intensifying racial inequalities. This injustice is rooted in historical policies like government-initiated segregation and redlining practices, where the Federal Housing Administration (FHA) policies discriminated against minority-centered regions in the housing markets. While overtly discriminatory practices were outlawed by the 1960s, these adverse practices still affect minority communities which plague this country to this day.

Today, disparities still persist where especially Black and Hispanic groups have lower home ownership rates opposed to majority communities. Government policies like the mortgage interest deduction also disproportionately benefit wealthy white homeowners, offering "ten times more benefits to those with incomes over \$250,000 than those with incomes below \$75,000." Similarly, "capital gains tax deductions on home sales... disproportionately benefit wealthy white homeowners," as minority homeowners are less likely to sell properties for profit due to lower appreciation rates in historically redlined neighborhoods.

Predatory lending practices exacerbate these inequities: "lenders, even accounting for income, are significantly more likely to target blacks and Hispanics with [subpar] loans." For example, "upper-income blacks are 360\% more likely to have subprime loans than upper-income whites," leading to higher foreclosure rates and wealth stripping. Subprime refinancing—"nearly three times as likely for blacks compared to whites"—further erodes home equity, perpetuating cycles of poverty. 

Our project addresses these injustices by analyzing mortgage lending data in Rhode Island to expose systemic biases. Using the Home Mortgage Disclosure Act (HMDA) dataset, we will aim to identify discrepancies in lending practice by examining variables such as applicant race, geography, ethnicity, income, loan amounts, and approval/denial rates. Our work in exposing the systemic biases encoded within financial systems will empower affected communities and influence policymakers to address these disparities. Therefore, our work aligns with contemporary discussions of data justice by challenging biased financial systems and helping to dismantle structural inequalities via a technical and data science approach.

We use features
- Loan to value ratio
- Applicant Race
- Applicant Ethnicity
- Applicant Sex
- Applicant income
- Loan type
- Property type
- Interest rate
- Lien status
- Loan purpose
- Occupancy type
- Debt to income ratio

To predict the target variable: **approval status of the loan**. The target variable is binary, where 1 indicates that the loan was approved and 0 indicates that it was not.

We first conduct an exploratory data analysis to understand the data and its distributions. We then build a logistic regression model to predict the approval status of the loan based on the features. Finally, we build an XGBoost model to improve the accuracy of our predictions. The models are evaluated using metrics such as accuracy, precision, recall, and F1 score.
    """)
elif section == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("""
    This dashboard provides interactive visualizations to explore mortgage approval data.
    Each graph is designed to help you understand key trends and disparities in the lending process,
    offering insights into how different factors intersect with social justice concerns.
    """)

    # Approval Rates by Race (Interactive)
    approval_by_race = df.groupby('derived_race')[
        'loan_approved'].mean().reset_index()
    fig_race = px.bar(
        approval_by_race,
        x='derived_race',
        y='loan_approved',
        title="Approval Rates by Race",
        labels={'derived_race': "Race", 'loan_approved': "Approval Rate"},
        hover_data={'loan_approved': ':.2f'}
    )
    st.markdown(
        "This bar chart displays the average mortgage approval rates segmented by race. We can see that underrepresented groups such as Black and Hispanic applicants have lower approval rates compared to their white counterparts.")
    st.plotly_chart(fig_race, use_container_width=True)

    # Approval Rates by Ethnicity (Interactive)
    approval_by_ethnicity = df.groupby('derived_ethnicity')[
        'loan_approved'].mean().reset_index()
    fig_ethnicity = px.bar(
        approval_by_ethnicity,
        x='derived_ethnicity',
        y='loan_approved',
        title="Approval Rates by Ethnicity",
        labels={'derived_ethnicity': "Ethnicity",
                'loan_approved': "Approval Rate"},
        hover_data={'loan_approved': ':.2f'}
    )
    st.markdown(
        "This bar chart displays the average mortgage approval rates segmented by ethnicity. We can see that underrepresented groups such as Hispanic applicants have lower approval rates compared to their white counterparts.")
    st.plotly_chart(fig_ethnicity, use_container_width=True)

    # Loan Amount Distributions by Race (Interactive Histogram with Box Plot)
    fig_hist_race = px.histogram(
        df,
        x='loan_amount',
        color='derived_race',
        marginal='box',
        barmode='overlay',
        title="Loan Amount Distribution by Race",
        labels={'loan_amount': "Loan Amount", 'count': "Frequency"}
    )
    st.markdown(
        "This histogram with box plot overlay displays loan amount distributions across races.")
    st.plotly_chart(fig_hist_race, use_container_width=True)

    # Loan Amount Distributions by Ethnicity (Interactive Histogram with Box Plot)
    fig_hist_ethnicity = px.histogram(
        df,
        x='loan_amount',
        color='derived_ethnicity',
        marginal='box',
        barmode='overlay',
        title="Loan Amount Distribution by Ethnicity",
        labels={'loan_amount': "Loan Amount", 'count': "Frequency"}
    )
    st.markdown(
        "This histogram with box plot overlay displays loan amount distributions across ethnicities.")
    st.plotly_chart(fig_hist_ethnicity, use_container_width=True)

    # Correlation Matrix (Interactive Heatmap)
    st.header("Correlation Matrix")
    # Create encoded columns for correlation
    df_corr = df.copy()
    df_corr['loan_approved_encoded'] = df_corr['loan_approved'].astype(
        'category').cat.codes
    df_corr['derived_race_encoded'] = df_corr['derived_race'].astype(
        'category').cat.codes
    cols = ['loan_approved_encoded', 'derived_race_encoded',
            'loan_to_value_ratio', 'debt_to_income_ratio']
    corr_matrix = df_corr[cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix",
        labels=dict(x="Variable", y="Variable", color="Correlation")
    )
    st.markdown(
        "This heatmap illustrates the correlation relationships among selected variables. Correlation values range from -1 to 1, where 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation. We see slight correlation between race to approval status, as well as debt to income ratio to loan to value ratio. This correlation is necessary for conducting regression analysis to prevent multicollinearity in the features.")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Choropleth Map (Already Interactive)
    st.header("Approval Rates by County")
    df['county_code'] = pd.to_numeric(df['county_code'], errors='coerce').fillna(
        0).astype(int).astype(str).str.zfill(5)
    df_grouped = df.groupby('county_code')[
        'loan_approved'].mean().reset_index()
    fig_choropleth = px.choropleth(
        df_grouped,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations='county_code',
        color='loan_approved',
        scope="usa",
        color_continuous_scale="Viridis",
        title="Approval Rates by County",
        labels={'loan_approved': "Approval Rate"}
    )

    # Update geos to zoom into Rhode Island (center set to RI coordinates)
    fig_choropleth.update_geos(
        visible=False,
        resolution=50,
        showcountries=True,
        countrycolor="Black",
        showsubunits=True,
        subunitcolor="Blue",
        center=dict(lat=41.7, lon=-71.5),  # RI approximate center
        projection_scale=35               # Adjust zoom level as needed
    )
    st.markdown(
        "This choropleth map displays the average mortgage approval rates by county with interactive zoom features. The approval rates do not seem to vary significantly by county, but we can see that the counties with the highest approval rates are in the southern part of the state. This may be due to the fact that these counties are more affluent and have a higher percentage of white residents.")
    st.plotly_chart(fig_choropleth, use_container_width=True)
elif section == "Logistic Regression Model":
    st.title("Logistic Regression Model")
    st.markdown("""
    Logistic Regression is used to model the binary outcome of mortgage approvals.
    The model's coefficients reveal how different features impact approval decisions,
    highlighting potential systemic biases and informing discussions on fairness and social justice.
    """)

    # Model training with caching
    @st.cache_resource
    def train_logreg():
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        return clf, X_test, y_test

    clf, X_test, y_test = train_logreg()

    # Evaluation
    st.header("Model Performance")
    y_pred = clf.predict(X_test)

    # Interactive Confusion Matrix using Plotly
    cm = confusion_matrix(y_test, y_pred)
    import plotly.graph_objects as go
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Denied", "Approved"],
        y=["Denied", "Approved"],
        colorscale='Blues',
        hoverongaps=False,
        text=cm,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig_cm.update_layout(
        title="Logistic Regression Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True"
    )
    st.markdown(
        "This heatmap represents the confusion matrix of the Logistic Regression model, comparing true and predicted classes. It seems that the model is slightly biased towards approving loans, and makes mistakes on approved vs denied loans when they were actually denied.")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report")
    st.markdown(
        "Below is the detailed classification report for the Logistic Regression model.")
    st.code(classification_report(y_test, y_pred))

    # Coefficients
    # st.header("Feature Coefficients")
    logreg = clf.named_steps['classifier']
    ohe = clf.named_steps['preprocessor'].named_transformers_['cat']
    all_features = np.concatenate(
        [numeric_cols, ohe.get_feature_names_out(categorical_cols)])
    coeff_df = pd.DataFrame(
        {'Feature': all_features, 'Coefficient': logreg.coef_[0]})
    # st.markdown("This table shows the coefficients of each feature, indicating their impact on mortgage approval in the Logistic Regression model.")
    # st.dataframe(coeff_df.sort_values(
    #     by='Coefficient', key=abs, ascending=False))

    # Interactive Plot for Feature Importances (sorted low to high)
    st.header("% Impact of Feature on Approval Rate, Ceteris Paribus")
    coeff_df_sorted = coeff_df.sort_values(by='Coefficient', ascending=True)
    colors = coeff_df_sorted['Coefficient'].apply(
        lambda x: 'blue' if x > 0 else 'red').tolist()
    fig_bar = go.Figure(go.Bar(
        x=coeff_df_sorted['Coefficient'],
        y=coeff_df_sorted['Feature'],
        orientation='h',
        marker=dict(color=colors),
        hovertemplate='%{y}: %{x}<extra></extra>'
    ))
    fig_bar.update_layout(
        title="Logistic Regression Feature Importances (Low to High)",
        xaxis_title="Coefficient",
        yaxis_title="Feature"
    )
    st.markdown(
        "This bar chart visualizes the impact of each feature on the approval rate, ceteris paribus.")
    st.plotly_chart(fig_bar, use_container_width=True)
elif section == "XGBoost Model":
    st.title("XGBoost Model")
    st.markdown("""
    XGBoost uses ensemble learning to capture complex relationships in the data.
    Through detailed feature importances and SHAP analysis, this section explains
    the multifaceted factors influencing mortgage decisions, aiding in the evaluation of social equity in lending.
    """)

    # Reuse preprocessing from Logistic Regression
    features = ['derived_ethnicity', 'derived_race', 'derived_sex', 'loan_type',
                'loan_purpose', 'lien_status', 'loan_amount', 'loan_to_value_ratio',
                'interest_rate', 'property_value', 'occupancy_type', 'debt_to_income_ratio']
    target = 'loan_approved'

    X = df[features]
    y = df[target].astype('category').cat.codes

    # Model training with caching
    @st.cache_resource
    def train_xgb():
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                use_label_encoder=False, eval_metric='logloss'))
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        return clf, X_test, y_test

    clf, X_test, y_test = train_xgb()

    # Generate proper feature names using the preprocessor inside the pipeline
    ohe = clf.named_steps['preprocessor'].named_transformers_['cat']
    all_features = np.concatenate(
        [numeric_cols, ohe.get_feature_names_out(categorical_cols)])

    # Evaluation
    st.header("Model Performance")
    y_pred = clf.predict(X_test)

    # Interactive Confusion Matrix using Plotly
    cm = confusion_matrix(y_test, y_pred)
    import plotly.graph_objects as go
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Denied", "Approved"],
        y=["Denied", "Approved"],
        colorscale='Blues',
        hoverongaps=False,
        text=cm,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig_cm.update_layout(
        title="XGBoost Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True"
    )
    st.markdown(
        "This heatmap represents the confusion matrix of the XGBoost model, comparing actual versus predicted outcomes.")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report")
    st.markdown("Below is the classification report for the XGBoost model.")
    st.code(classification_report(y_test, y_pred))

    # Feature Importance
    st.header("Feature Importance")
    xgb_model = clf.named_steps['classifier']
    importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Interactive Plot for Top 20 Feature Importances using Plotly
    fig_fi = go.Figure(go.Bar(
        x=feature_importance_df.head(20)['Importance'],
        y=feature_importance_df.head(20)['Feature'],
        orientation='h',
        marker=dict(color='royalblue'),
        hovertemplate='%{y}: %{x}<extra></extra>'
    ))
    fig_fi.update_layout(
        title="XGBoost Feature Importance (Top 20)",
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    st.markdown(
        "This bar chart displays the top 20 most important features as identified by the XGBoost model.")
    st.plotly_chart(fig_fi, use_container_width=True)

    # SHAP values
    st.header("SHAP Summary Plot")
    preprocessor = clf.named_steps['preprocessor']
    X_test_preprocessed = preprocessor.transform(X_test)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_preprocessed)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test_preprocessed,
                      feature_names=all_features, show=False)
    st.markdown(
        "The SHAP summary plot illustrates the feature impact on model output for the XGBoost model.")
    st.pyplot(fig)
