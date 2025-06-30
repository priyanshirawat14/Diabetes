import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="🩺 Diabetes Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

df = load_data()

st.title("🩺 Diabetes Health Indicators Dashboard (BRFSS 2015)")
st.markdown("This dashboard provides macro and micro-level visual insights to help HR & healthcare leaders understand diabetes trends.")

# Sidebar filters
st.sidebar.header("📌 Filter the Dataset")
age_slider = st.sidebar.slider("Select Age Code Range", int(df['Age'].min()), int(df['Age'].max()), (30, 70))
sex_option = st.sidebar.radio("Select Sex", ["All", "Male", "Female"])

filtered_df = df[(df['Age'] >= age_slider[0]) & (df['Age'] <= age_slider[1])]
if sex_option != "All":
    filtered_df = filtered_df[filtered_df['Sex'] == (1 if sex_option == "Male" else 0)]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Detailed Insights", "📂 Raw Data"])

# === Overview Tab ===
with tab1:
    st.subheader("1. Diabetes Cases Overview")
    st.markdown("Shows the count of individuals with and without diabetes.")
    diabetes_counts = filtered_df['Diabetes_binary'].value_counts().rename({0: "No Diabetes", 1: "Diabetes"})
    st.bar_chart(diabetes_counts)

    st.subheader("2. Age Distribution by Diabetes")
    st.markdown("Compares age distribution of diabetic vs. non-diabetic individuals.")
    fig1, ax1 = plt.subplots()
    sns.histplot(data=filtered_df, x="Age", hue="Diabetes_binary", kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("3. General Health Status vs Diabetes")
    st.markdown("General self-reported health by diabetes status.")
    fig2 = px.histogram(filtered_df, x="GenHlth", color="Diabetes_binary", barmode="group")
    st.plotly_chart(fig2)

    st.subheader("4. BMI Distribution by Diabetes")
    st.markdown("Box plot comparing BMI of diabetic and non-diabetic groups.")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Diabetes_binary", y="BMI", data=filtered_df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("5. Gender-wise Diabetes Percentage")
    st.markdown("Average diabetes rate among males and females.")
    sex_grouped = filtered_df.groupby("Sex")["Diabetes_binary"].mean().rename({0: "Female", 1: "Male"})
    st.bar_chart(sex_grouped)

# === Detailed Insights Tab ===
with tab2:
    st.subheader("6. Physical Activity vs Diabetes")
    st.markdown("Impact of regular physical activity on diabetes prevalence.")
    st.plotly_chart(px.histogram(filtered_df, x="PhysActivity", color="Diabetes_binary", barmode="group"))

    st.subheader("7. Fruit Consumption vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="Fruits", color="Diabetes_binary", barmode="group"))

    st.subheader("8. Vegetable Intake vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="Veggies", color="Diabetes_binary", barmode="group"))

    st.subheader("9. Smoking Status vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="Smoker", color="Diabetes_binary", barmode="group"))

    st.subheader("10. High Blood Pressure vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="HighBP", color="Diabetes_binary", barmode="group"))

    st.subheader("11. High Cholesterol vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="HighChol", color="Diabetes_binary", barmode="group"))

    st.subheader("12. Healthcare Access vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="NoDocbcCost", color="Diabetes_binary", barmode="group"))

    st.subheader("13. Cholesterol Check vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="CholCheck", color="Diabetes_binary", barmode="group"))

    st.subheader("14. Sleep Impact (MentHlth Days)")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x="Diabetes_binary", y="MentHlth", data=filtered_df, ax=ax4)
    st.pyplot(fig4)

    st.subheader("15. Physical Health Days vs Diabetes")
    st.plotly_chart(px.box(filtered_df, x="Diabetes_binary", y="PhysHlth", color="Diabetes_binary"))

    st.subheader("16. Education Level vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="Education", color="Diabetes_binary", barmode="group"))

    st.subheader("17. Income Level vs Diabetes")
    st.plotly_chart(px.histogram(filtered_df, x="Income", color="Diabetes_binary", barmode="group"))

    st.subheader("18. Diabetes Rate by Age Group (Bins)")
    age_bins = pd.cut(filtered_df['Age'], bins=[18, 30, 40, 50, 60, 70, 80])
    st.plotly_chart(px.histogram(filtered_df, x=age_bins, color="Diabetes_binary", barmode="group"))

    st.subheader("19. BMI vs Mental Health (Diabetes Status)")
    fig5 = px.scatter(filtered_df, x="BMI", y="MentHlth", color="Diabetes_binary")
    st.plotly_chart(fig5)

    st.subheader("20. Heatmap: Correlation Matrix")
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    sns.heatmap(filtered_df.corr(), cmap="coolwarm", annot=False, ax=ax6)
    st.pyplot(fig6)

# === Raw Data Tab ===
with tab3:
    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df)
