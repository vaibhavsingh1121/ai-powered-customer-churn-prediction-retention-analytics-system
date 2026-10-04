import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained model relative to this file, independent of the launch directory.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "customer_churn_model.joblib"
model = joblib.load(MODEL_PATH)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Executive Dashboard",
        "Churn Prediction",
        "Batch Prediction",
        "About"
    ]
)

# Home page
if page == "Home":
    st.title("Customer Churn Prediction System")
    st.subheader("AI-Powered Customer Retention Analytics")

    st.write("""
    This application uses Machine Learning to identify customers
    who are likely to discontinue a service.

    **Key Features**
    - Customer churn prediction
    - Churn probability estimation
    - Data-driven customer retention insights
    """)

# Executive Dashboard
elif page == "Executive Dashboard":
    st.title("Executive Analytics Dashboard")
    st.write("Customer churn overview and business insights")

    # Load the dataset relative to this file.
    DATA_PATH = APP_DIR / "dataset" / "Telco-Customer-Churn.csv"

    df = pd.read_csv(DATA_PATH)

    # Dashboard Filters
    st.markdown("---")
    st.subheader("Filter Customer Segments")

    filter1, filter2, filter3 = st.columns(3)

    with filter1:
        selected_gender = st.selectbox(
            "Gender",
            ["All"] + sorted(df["gender"].unique().tolist())
        )

    with filter2:
        selected_contract = st.selectbox(
            "Contract Type",
            ["All"] + sorted(df["Contract"].unique().tolist())
        )

    with filter3:
        selected_internet = st.selectbox(
            "Internet Service",
            ["All"] + sorted(df["InternetService"].unique().tolist())
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_gender != "All":
        filtered_df = filtered_df[
            filtered_df["gender"] == selected_gender
        ]

    if selected_contract != "All":
        filtered_df = filtered_df[
            filtered_df["Contract"] == selected_contract
        ]

    if selected_internet != "All":
        filtered_df = filtered_df[
            filtered_df["InternetService"] == selected_internet
        ]

    if filtered_df.empty:
        st.info("No customers match the selected filters.")
    else:
        # Calculate KPIs using filtered data
        total_customers = len(filtered_df)
        churned_customers = (filtered_df["Churn"] == "Yes").sum()

        churn_rate = (
            (churned_customers / total_customers) * 100
            if total_customers > 0 else 0
        )

        avg_monthly_charges = (
            filtered_df["MonthlyCharges"].mean()
            if total_customers > 0 else 0
        )

        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churned Customers", f"{churned_customers:,}")
        col3.metric("Churn Rate", f"{churn_rate:.2f}%")
        col4.metric("Avg. Monthly Charges", f"${avg_monthly_charges:.2f}")

        st.markdown("---")
        st.subheader("Customer Churn Analysis")

        # Create two columns for charts
        chart1, chart2 = st.columns([1, 1], gap="large")

        # Churn Distribution
        with chart1:
            churn_counts = filtered_df["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Churn", "Customers"]

            fig1 = px.pie(
                churn_counts,
                names="Churn",
                values="Customers",
                title="Churn Distribution",
                hole=0.55,
                color="Churn",
                color_discrete_map={
                    "Yes": "#EF4444",
                    "No": "#10B981"
                }
            )

            fig1.update_traces(
                textinfo="percent+label",
                textposition="inside",
                insidetextfont=dict(color="white", size=13),
                hovertemplate="%{label}: %{value} customers<extra></extra>"
            )

            fig1.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(
                fig1,
                width="stretch",
                key="churn_distribution_chart"
            )

        # Churn Analysis by Payment Method
        st.markdown("---")
        st.subheader("Churn Analysis by Payment Method")

        payment_churn = (
            filtered_df.groupby("PaymentMethod", as_index=False)
            .agg(
                Total_Customers=("Churn", "count"),
                Churned_Customers=(
                    "Churn",
                    lambda x: (x == "Yes").sum()
                )
            )
        )

        payment_churn["Churn_Rate"] = (
            payment_churn["Churned_Customers"]
            / payment_churn["Total_Customers"].replace(0, 1)
            * 100
        )

        payment_churn = payment_churn.sort_values(
            "Churn_Rate",
            ascending=True
        )

        fig4 = px.bar(
            payment_churn,
            x="Churn_Rate",
            y="PaymentMethod",
            orientation="h",
            title="Churn Rate by Payment Method",
            text=payment_churn["Churn_Rate"].round(1),
            labels={
                "PaymentMethod": "Payment Method",
                "Churn_Rate": "Churn Rate (%)"
            },
            color="Churn_Rate",
            color_continuous_scale="Oranges"
        )

        fig4.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
            cliponaxis=False
        )

        fig4.update_layout(
            height=450,
            xaxis=dict(
                title="Churn Rate (%)",
                range=[0, 110]
            ),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            margin=dict(l=20, r=40, t=60, b=30)
        )

        st.plotly_chart(
            fig4,
            use_container_width=True,
            key="payment_method_churn_chart"
        )

        # Key Business Insights
        st.markdown("---")
        st.subheader("Key Business Insights")

        # Contract churn analysis
        contract_insights = (
            filtered_df.groupby("Contract")
            .agg(
                Total_Customers=("Churn", "count"),
                Churned_Customers=("Churn", lambda x: (x == "Yes").sum())
            )
            .reset_index()
        )

        contract_insights["Churn_Rate"] = (
            contract_insights["Churned_Customers"]
            / contract_insights["Total_Customers"].replace(0, 1) * 100
        )

        # Payment method churn analysis
        payment_insights = (
            filtered_df.groupby("PaymentMethod")
            .agg(
                Total_Customers=("Churn", "count"),
                Churned_Customers=("Churn", lambda x: (x == "Yes").sum())
            )
            .reset_index()
        )

        payment_insights["Churn_Rate"] = (
            payment_insights["Churned_Customers"]
            / payment_insights["Total_Customers"].replace(0, 1) * 100
        )

        if not contract_insights.empty and not payment_insights.empty:

            highest_contract = contract_insights.iloc[
                contract_insights["Churn_Rate"].argmax()
            ]

            lowest_contract = contract_insights.iloc[
                contract_insights["Churn_Rate"].argmin()
            ]

            highest_payment = payment_insights.iloc[
                payment_insights["Churn_Rate"].argmax()
            ]

            lowest_payment = payment_insights.iloc[
                payment_insights["Churn_Rate"].argmin()
            ]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Highest Contract Churn",
                f"{highest_contract['Churn_Rate']:.1f}%",
                help=highest_contract["Contract"]
            )

            col2.metric(
                "Lowest Contract Churn",
                f"{lowest_contract['Churn_Rate']:.1f}%",
                help=lowest_contract["Contract"]
            )

            col3.metric(
                "Highest Payment Churn",
                f"{highest_payment['Churn_Rate']:.1f}%",
                help=highest_payment["PaymentMethod"]
            )

            col4.metric(
                "Lowest Payment Churn",
                f"{lowest_payment['Churn_Rate']:.1f}%",
                help=lowest_payment["PaymentMethod"]
            )

            st.markdown("### Retention Recommendations")

            st.info(
                f"**Contract Strategy:** {highest_contract['Contract']} "
                f"customers have the highest churn rate "
                f"({highest_contract['Churn_Rate']:.1f}%). "
                "Consider personalized retention offers and incentives "
                "for longer-term contracts."
            )

            st.warning(
                f"**Payment Strategy:** Customers using "
                f"{highest_payment['PaymentMethod']} have the highest "
                f"payment-method churn rate "
                f"({highest_payment['Churn_Rate']:.1f}%). "
                "Investigate potential payment-related friction and "
                "promote convenient payment alternatives."
            )

            # Churn by Contract Type
            with chart2:
                contract_churn = (
                    filtered_df.groupby("Contract", as_index=False)
                    .agg(
                        Total_Customers=("Churn", "count"),
                        Churned_Customers=(
                            "Churn",
                            lambda x: (x == "Yes").sum()
                        )
                    )
                )

                contract_churn["Churn_Rate"] = (
                    contract_churn["Churned_Customers"]
                    / contract_churn["Total_Customers"].replace(0, 1)
                    * 100
                )

                fig2 = px.bar(
                    contract_churn,
                    x="Churn_Rate",
                    y="Contract",
                    orientation="h",
                    title="Churn Rate by Contract Type",
                    text=contract_churn["Churn_Rate"].round(1),
                    labels={
                        "Contract": "Contract Type",
                        "Churn_Rate": "Churn Rate (%)"
                    },
                    color="Churn_Rate",
                    color_continuous_scale="Reds"
                )

                fig2.update_traces(
                    texttemplate="%{text}%",
                    textposition="outside",
                    cliponaxis=False
                )

                fig2.update_layout(
                    height=400,
                    xaxis=dict(
                        title="Churn Rate (%)",
                        range=[0, 110]
                    ),
                    yaxis=dict(
                        title="",
                        categoryorder="total ascending"
                    ),
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=40, t=60, b=30)
                )

                st.plotly_chart(
                    fig2,
                    width="stretch",
                    key="contract_churn_chart"
                )

                # Churn by Tenure
                st.markdown("---")
                st.subheader("Customer Tenure Analysis")

                # Work with filtered data without modifying the original dataset
                tenure_df = filtered_df.copy()

                # Create tenure groups
                tenure_df["TenureGroup"] = pd.cut(
                    tenure_df["tenure"],
                    bins=[-1, 12, 24, 48, 72],
                    labels=[
                        "0–12 Months",
                        "13–24 Months",
                        "25–48 Months",
                        "49–72 Months"
                    ]
                )

                # Calculate churn by tenure group
                tenure_churn = (
                    tenure_df.groupby("TenureGroup", observed=False)
                    .agg(
                        Total_Customers=("Churn", "count"),
                        Churned_Customers=(
                            "Churn",
                            lambda x: (x == "Yes").sum()
                        )
                    )
                    .reset_index()
                )

                # Calculate churn rate
                tenure_churn["Churn_Rate"] = (
                    tenure_churn["Churned_Customers"]
                    / tenure_churn["Total_Customers"].replace(0, 1)
                    * 100
                )

                # Create tenure chart
                fig3 = px.bar(
                    tenure_churn,
                    x="TenureGroup",
                    y="Churn_Rate",
                    title="Churn Rate by Customer Tenure",
                    text=tenure_churn["Churn_Rate"].round(1),
                    labels={
                        "TenureGroup": "Customer Tenure",
                        "Churn_Rate": "Churn Rate (%)"
                    },
                    color="Churn_Rate",
                    color_continuous_scale="Oranges"
                )

                fig3.update_traces(
                    texttemplate="%{text}%",
                    textposition="outside"
                )

                fig3.update_layout(yaxis_range=[0, 100])

                st.plotly_chart(
                    fig3,
                    width="stretch",
                    key="tenure_churn_chart"
                )

# Churn prediction page
elif page == "Churn Prediction":
    st.title("Customer Churn Prediction")
    st.write("Enter the customer details below to predict churn.")

    # Customer information
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=float(tenure * monthly_charges)
        )

    # Predict churn
    if st.button("Predict Churn", type="primary"):
        customer_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        prediction = model.predict(customer_data)[0]
        probability = model.predict_proba(customer_data)[0][1]

    

        # Display prediction result
        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        # Risk classification
        if probability >= 0.70:
            risk_level = "High"
            st.error("High Churn Risk")
        elif probability >= 0.40:
            risk_level = "Medium"
            st.warning("Medium Churn Risk")
        else:
            risk_level = "Low"
            st.success("Low Churn Risk")

        st.progress(float(probability))

        # Retention recommendations
        st.markdown("---")
        st.subheader("Retention Recommendations")

        recommendations = []

        if probability >= 0.70:
            recommendations.append(
                "Prioritize this customer for a personalized retention campaign."
            )

        if contract == "Month-to-month":
            recommendations.append(
                "Offer an incentive to encourage a one-year or two-year contract."
            )

        if tenure <= 12:
            recommendations.append(
                "Provide onboarding support and early-stage engagement offers."
            )

        if monthly_charges >= 80:
            recommendations.append(
                "Review the customer's pricing and consider a suitable discount."
            )

        if online_security == "No" and internet_service != "No":
            recommendations.append(
                "Consider offering an online security package."
            )

        if tech_support == "No" and internet_service != "No":
            recommendations.append(
                "Promote a technical support plan to improve the customer experience."
            )

        if not recommendations:
            recommendations.append(
                "Continue regular engagement and monitor future churn risk."
            )

        for i, recommendation in enumerate(recommendations, start=1):
            st.write(f"{i}. {recommendation}")

# About page
elif page == "About":
    st.title("About the Project")

    st.write("""
    **Project:** AI-Powered Customer Churn Prediction
    & Retention Analytics System

    **Model:** Random Forest Classifier

    **Dataset:** IBM Telco Customer Churn Dataset

    **Objective:** Predict customer churn and support
    data-driven customer retention strategies.
    """)

elif page == "Batch Prediction":
    st.title("Batch Churn Prediction")
    st.write("Upload a CSV file to predict churn risk for multiple customers.")

    required_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]

    uploaded_file = st.file_uploader(
        "Upload Customer CSV File",
        type=["csv"],
        key="batch_file_uploader"
    )

    # Clear previous results when a new file is uploaded
    if uploaded_file is not None:
        file_signature = (
            uploaded_file.name,
            uploaded_file.size
        )

        if st.session_state.get("batch_file_signature") != file_signature:
            st.session_state["batch_file_signature"] = file_signature
            st.session_state["batch_predictions_ready"] = False
            st.session_state.pop("batch_results", None)

        try:
            batch_df = pd.read_csv(uploaded_file)

            # Validate required columns
            missing_columns = [
                col for col in required_columns
                if col not in batch_df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )
                st.info(
                    "Please upload a CSV file containing all "
                    "required customer features."
                )

            elif batch_df.empty:
                st.error("The uploaded CSV file contains no customer records.")

            else:
                st.success(
                    f"File uploaded successfully! "
                    f"Total records: {len(batch_df):,}"
                )

                st.subheader("Uploaded Data")
                st.dataframe(batch_df, width="stretch")

                if st.button("Predict Churn", type="primary"):
                    prediction_df = batch_df.drop(
                        columns=["customerID", "Churn"],
                        errors="ignore"
                    ).copy()

                    numeric_columns = [
                        "SeniorCitizen",
                        "tenure",
                        "MonthlyCharges",
                        "TotalCharges"
                    ]

                    invalid_data = False

                    for col in numeric_columns:
                        prediction_df[col] = pd.to_numeric(
                            prediction_df[col],
                            errors="coerce"
                        )

                        if prediction_df[col].isnull().any():
                            st.error(
                                f"Invalid or missing values in '{col}'. "
                                "Please correct the CSV file."
                            )
                            invalid_data = True

                    if not invalid_data:
                        try:
                            predictions = model.predict(prediction_df)
                            probabilities = model.predict_proba(
                                prediction_df
                            )[:, 1]

                            results = batch_df.copy()

                            results["Churn Prediction"] = [
                                "Yes" if pred == 1 else "No"
                                for pred in predictions
                            ]

                            results["Churn Probability (%)"] = (
                                probabilities * 100
                            ).round(2)

                            results["Risk Level"] = [
                                "High" if prob >= 0.70
                                else "Medium" if prob >= 0.40
                                else "Low"
                                for prob in probabilities
                            ]

                            st.session_state["batch_results"] = results
                            st.session_state["batch_predictions_ready"] = True

                        except Exception as e:
                            st.session_state["batch_predictions_ready"] = False
                            st.session_state.pop("batch_results", None)
                            st.error(f"Prediction failed: {e}")

                # Display results only after successful prediction
                if st.session_state.get("batch_predictions_ready", False):
                    batch_results = st.session_state["batch_results"]

                    total_customers = len(batch_results)
                    predicted_churners = (
                        batch_results["Churn Prediction"] == "Yes"
                    ).sum()
                    high_risk_customers = (
                        batch_results["Risk Level"] == "High"
                    ).sum()
                    avg_churn_probability = (
                        batch_results["Churn Probability (%)"].mean()
                    )

                    st.subheader("Batch Prediction Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Total Customers", total_customers)
                    col2.metric("Predicted Churners", predicted_churners)
                    col3.metric("High-Risk Customers", high_risk_customers)
                    col4.metric(
                        "Avg. Churn Probability",
                        f"{avg_churn_probability:.2f}%"
                    )

                    st.subheader("Customer Risk Distribution")

                    risk_counts = (
                        batch_results["Risk Level"]
                        .value_counts()
                        .reindex(
                            ["Low", "Medium", "High"],
                            fill_value=0
                        )
                    )

                    fig = px.pie(
                        names=risk_counts.index,
                        values=risk_counts.values,
                        title="Distribution of Customer Risk Levels",
                        hole=0.4,
                        color=risk_counts.index,
                        color_discrete_map={
                            "Low": "#2ecc71",
                            "Medium": "#f39c12",
                            "High": "#e74c3c"
                        }
                    )

                    fig.update_traces(
                        textinfo="percent+label",
                        hovertemplate=(
                            "%{label}: %{value} customers<extra></extra>"
                        )
                    )

                    st.plotly_chart(fig, width="stretch")

                    st.subheader("Batch Prediction Results")

                    risk_filter = st.multiselect(
                        "Filter by Risk Level",
                        ["Low", "Medium", "High"],
                        default=["Low", "Medium", "High"],
                        key="batch_risk_filter"
                    )

                    filtered_results = batch_results[
                        batch_results["Risk Level"].isin(risk_filter)
                    ]

                    st.write(
                        f"Showing {len(filtered_results):,} "
                        f"of {len(batch_results):,} customers"
                    )

                    st.dataframe(
                        filtered_results,
                        width="stretch"
                    )

                    csv = batch_results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Prediction Results",
                        data=csv,
                        file_name="churn_prediction_results.csv",
                        mime="text/csv"
                    )

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")

        except Exception as e:
            st.error(f"Unable to process the uploaded CSV file: {e}")