# )
import numpy as np
import pandas as pd
import joblib
import ast  # Import ast to safely convert string to list
import streamlit as st
import base64
from fpdf import FPDF

precaution_df = pd.read_csv("symptom_precaution.csv")
medication_df = pd.read_csv("medications.csv")

# 🎯 Load Trained Model & Symptoms List
rf = joblib.load("disease_prediction_model.pkl")
all_symptoms = joblib.load("all_symptoms.pkl")
inverse_mapping = joblib.load("inverse_label_mapping.pkl")

# Streamlit UI
st.set_page_config(page_title="Disease Prediction System")
st.markdown(
    """
    <style>
        .card {
            background-color: rgba(104, 159, 255, 0.8);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.5);
            transition: transform 0.2s ease-in-out;
        }
        .card:hover {
            transform: scale(1.02);
        }
        .button-link {
            display: inline-block;
            padding: 10px 15px;
            margin-top: 20px;
            color: #ffffff;
            background-color: #fcffff;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
        }
        .button-link:hover {
            background-color: #fcffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Disease Prediction System 🧑🏻‍⚕️")
st.write("⚕️Your go-to Health Companion")
st.markdown("**🧔🏻Patient details:**")


def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function with your image file
set_bg("healthcare.jpeg")

st.markdown(
    """
    <style>
        .main {
            background: rgba(10, 10, 10, 0.7); /* White transparent background */
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ✅ User Symptoms Input
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []

symptom = st.sidebar.selectbox("Choose a Symptom: ", ["Select a symptom"] + all_symptoms)
if symptom != "Select a symptom" and symptom not in st.session_state.selected_symptoms:
    st.session_state.selected_symptoms.append(symptom)

st.sidebar.write("## Selected Symptoms:")
for s in st.session_state.selected_symptoms:
    # st.sidebar.write(f"✅ {s}")
    st.sidebar.markdown(f"<span style='color: red;'>✅ {s}</span>", unsafe_allow_html=True)  # Change text color to blue

if st.sidebar.button("Clear Symptoms", key="clear_btn"):
    st.session_state.selected_symptoms = []

# Custom CSS to only change text color of "Clear Symptoms" button
st.sidebar.markdown(
    """
    <style>
        /* Target the "Clear Symptoms" button specifically */
        div[data-testid="stButton"] button:nth-of-type(1) {
            color: white !important;  /* Change text color to red */
            font-weight: bold !important;  /* Make text bold */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 📚 Button to Open Notebook in Sidebar
st.sidebar.markdown(
    """
    <a href="https://colab.research.google.com/drive/1bMBiMGMK2Tjfdz49ry1ePT1hwdzXQNat?usp=sharing" class="button-link" target="_blank" style="color: white !important;">
        Open Training Notebook
    </a>
    """,
    unsafe_allow_html=True,
)


# Function to generate a well-formatted PDF Report
def generate_report(name, age, gender, mobile, symptoms, disease, precautions, medications):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 10, "DISEASE REPORT", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Patient Information:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Name: {name}", ln=True)
    pdf.cell(200, 10, f"Age: {age}", ln=True)
    pdf.cell(200, 10, f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, f"Mobile: {mobile}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Report Details:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Predicted Disease: {disease}")
    pdf.multi_cell(0, 10, f"Symptoms: {', '.join(symptoms)}")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Recommended Precautions:", ln=True)
    pdf.set_font("Arial", "", 12)
    for p in precautions:
        pdf.multi_cell(0, 10, f"- {p}")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Suggested Medications:", ln=True)
    pdf.set_font("Arial", "", 12)
    for med in medications:
        pdf.multi_cell(0, 10, f"- {med}")

    pdf.output("Disease_Report.pdf")
    return "Disease_Report.pdf"


# Collect Patient Details
name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)
gender = st.selectbox("Select your gender:", ["Male", "Female"])
mobile = st.text_input("Enter your mobile number:")

if st.button("Predict Disease"):
    if len(st.session_state.selected_symptoms) == 0:
        st.error("Please select at least one symptom!")
    elif not name or not age or not gender or not mobile:
        st.error("Please enter all details including name, age, gender, and mobile number!")
    else:
        user_input = np.zeros(len(all_symptoms))
        for symptom in st.session_state.selected_symptoms:
            if symptom in all_symptoms:
                idx = all_symptoms.index(symptom)
                user_input[idx] = 1

        user_input = np.array(user_input).reshape(1, -1)
        encoded_prediction = rf.predict(user_input)[0]
        predicted_disease = inverse_mapping.get(encoded_prediction, "Unknown Disease")

        # 🎯 Fetch Precautions
        precautions = precaution_df[precaution_df["Disease"] == predicted_disease].iloc[:, 1:].values.flatten()
        precautions = [p for p in precautions if pd.notna(p)]  # Remove NaN values

        # 🎯 Fetch Medications
        medications = medication_df[medication_df["Disease"] == predicted_disease]["Medication"].values
        if len(medications) > 0:
            medications = ast.literal_eval(medications[0])  # Convert string list to actual list

        # ✅ Display Results
        st.success(f"Predicted Disease: **{predicted_disease}**")

        if precautions:
                st.markdown(
                        f"""
                        <div class='card'>
                            <h3 style="color: white;">🧬 Recommended Precautions:</h3>
                            {''.join([f'<p style="color: white;">✅ {p}</p>' for p in precautions])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if medications:
                st.markdown(
                        f"""
                        <div class='card' style="margin-top: 20px;">
                            <h3 style="color: white;">💊 Suggested Medications:</h3>
                            {''.join([f'<p style="color: white;">💊 {med}</p>' for med in medications])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


        # Generate and Download Report
        report_file = generate_report(name, age, gender, mobile, st.session_state.selected_symptoms, predicted_disease,
                                      precautions, medications)
        with open(report_file, "rb") as f:
            st.write("")
            st.write("")
            st.write("")
            st.download_button("📥 Download Report", f, file_name="Disease_Report.pdf", mime="application/pdf")


# Model Details
model_name = "Random Forest Classifier"
model_accuracy = 97.5

# Add Footer (Fixed at Bottom)
st.markdown(
    f"""
    <style>
        .footer {{
            background-color: #689fff;  /* Pure White Background */
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-size: 14px;
            border-top: 2px solid #ccc; /* Light Grey Border on Top */
        }}
    </style>
    <div class="footer">
        <p><strong>📊 Model Used:</strong> {model_name}  |   <strong> 🔍 Accuracy:</strong> { model_accuracy}%</p>
    </div>
    """,
    unsafe_allow_html=True,
)