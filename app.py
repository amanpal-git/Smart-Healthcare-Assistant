# import numpy as np
# import pandas as pd
# import joblib
# import ast  # Import ast to safely convert string to list
# import streamlit as st
#
# precaution_df = pd.read_csv("symptom_precaution.csv")
# medication_df = pd.read_csv("medications.csv")
#
# #  Load Trained Model & Symptoms List
# rf = joblib.load("disease_prediction_model.pkl")
# all_symptoms = joblib.load("all_symptoms.pkl")
# inverse_mapping = joblib.load("inverse_label_mapping.pkl")
#
# # Streamlit UI
# st.set_page_config(page_title="Disease Prediction System", layout="wide")
# st.markdown(
#     """
#     <style>
#         [data-testid="stSidebar"] {
#             padding: 20px;
#             border-radius: 10px;
#             color: white;
#         }
#         [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
#             color: white;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
#
# st.title("Disease Prediction System 🧑‍⚕️")
# st.write("Your Health Tracker:")
#
# #  Initialize session state for selected symptoms
# if "selected_symptoms" not in st.session_state:
#     st.session_state.selected_symptoms = []
#
# #  User Symptoms Input
# symptom = st.sidebar.selectbox("Choose a Symptom: ", ["Select a symptom"] + all_symptoms)
#
# if symptom != "Select a symptom" and symptom not in st.session_state.selected_symptoms:
#     st.session_state.selected_symptoms.append(symptom)
#
# #  Display selected symptoms
# st.sidebar.write("### Selected Symptoms:")
# for s in st.session_state.selected_symptoms:
#     st.sidebar.write(f"✅ {s}")
#
# #  Button to clear selected symptoms
# if st.sidebar.button("Clear Symptoms"):
#     st.session_state.selected_symptoms = []
#
# #  Predict Button
# if st.button("Predict Disease"):
#     if len(st.session_state.selected_symptoms) == 0:
#         st.error("Please select at least one symptom!")
#     else:
#         user_input = np.zeros(len(all_symptoms))
#         for symptom in st.session_state.selected_symptoms:
#             if symptom in all_symptoms:
#                 idx = all_symptoms.index(symptom)
#                 user_input[idx] = 1
#
#         user_input = np.array(user_input).reshape(1,-1)
#         encoded_prediction = rf.predict(user_input)[0]
#         predicted_disease = inverse_mapping.get(encoded_prediction, "Unknown Disease")
#
#         #  Fetch Precautions
#         precautions = precaution_df[precaution_df["Disease"] == predicted_disease].iloc[:, 1:].values.flatten()
#         precautions = [p for p in precautions if pd.notna(p)]  # Remove NaN values
#
#         #  Fetch Medications
#         medications = medication_df[medication_df["Disease"] == predicted_disease]["Medication"].values
#         if len(medications) > 0:
#             medications = ast.literal_eval(medications[0])  # Convert string list to actual list
#
#         # Display Results
#         st.success(f"Predicted Disease: **{predicted_disease}**")
#
#         if precautions:
#             st.markdown(
#                 f"""
#                 <div style="background-color: rgba(50, 50, 50, 0.85); padding: 15px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.5);">
#                     <h3 style="color: white;">🩺 Recommended Precautions:</h3>
#                     {''.join([f'<p style="color: white;">✅ {p}</p>' for p in precautions])}
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#
#         if medications:
#             st.markdown(
#                 f"""
#                 <div style="background-color: rgba(50, 50, 50, 0.85); padding: 15px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.5); margin-top: 20px;">
#                     <h3 style="color: white;">💊 Suggested Medications:</h3>
#                     {''.join([f'<p style="color: white;">💊 {med}</p>' for med in medications])}
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#
#
# # Model Details
# model_name = "Random Forest Classifier"
# model_accuracy = 97.5
#
# # Add Footer (Fixed at Bottom)
# st.markdown(
#     f"""
#     <style>
#         .footer {{
#             position: fixed;
#             bottom: 0;
#             left: 0;
#             width: 100%;
#             text-align: center;
#             padding: 10px;
#             background: rgba(50, 50, 50, 100)
#         }}
#     </style>
#     <div class="footer">
#         <p><strong>📊 Model Used:</strong> {model_name}  |   <strong> 🔍 Accuracy:</strong> { model_accuracy}%</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
import numpy as np
import pandas as pd
import joblib
import ast  # Import ast to safely convert string to list
import streamlit as st

precaution_df = pd.read_csv("symptom_precaution.csv")
medication_df = pd.read_csv("medications.csv")

# 🎯 Load Trained Model & Symptoms List
rf = joblib.load("disease_prediction_model.pkl")
all_symptoms = joblib.load("all_symptoms.pkl")
inverse_mapping = joblib.load("inverse_label_mapping.pkl")

# Streamlit UI
st.set_page_config(page_title="Disease Prediction System", layout="wide")
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #1e1e1e, #3e3e3e);
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
            color: white;
        }
        .card {
            background-color: rgba(50, 50, 50, 0.85);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
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
            background-color: #0078ff;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
        }
        .button-link:hover {
            background-color: #005fcc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Disease Prediction System 🧑‍⚕️")
st.write("Select the symptoms you are experiencing:")

# ✅ User Symptoms Input
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []

symptom = st.sidebar.selectbox("Choose a Symptom: ", ["Select a symptom"] + all_symptoms)
if symptom != "Select a symptom" and symptom not in st.session_state.selected_symptoms:
    st.session_state.selected_symptoms.append(symptom)

st.sidebar.write("### Selected Symptoms:")
for s in st.session_state.selected_symptoms:
    st.sidebar.write(f"✅ {s}")

if st.sidebar.button("Clear Symptoms", key="clear_btn"):
    st.session_state.selected_symptoms = []

# Custom CSS to only change text color of "Clear Symptoms" button
st.sidebar.markdown(
    """
    <style>
        /* Target the "Clear Symptoms" button specifically */
        div[data-testid="stButton"] button:nth-of-type(1) {
            color: red !important;  /* Change text color to red */
            font-weight: bold !important;  /* Make text bold */
        }
    </style>
    """,
    unsafe_allow_html=True,
)


#predict Button
if st.button("Predict Disease"):
    if len(st.session_state.selected_symptoms) == 0:
        st.error("Please select at least one symptom!")
    else:
        user_input = np.zeros(len(all_symptoms))
        for symptom in st.session_state.selected_symptoms:
            if symptom in all_symptoms:
                idx = all_symptoms.index(symptom)
                user_input[idx] = 1

        user_input = np.array(user_input).reshape(1,-1)
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

# 📚 Button to Open Notebook in Sidebar
st.sidebar.markdown(
    """
    <a href="https://colab.research.google.com/drive/your_notebook_link_here" class="button-link" target="_blank" style="color: white !important;">
        Open Training Notebook
    </a>
    """,
    unsafe_allow_html=True,
)
# Model Details
model_name = "Random Forest Classifier"
model_accuracy = 97.5

# Add Footer (Fixed at Bottom)
st.markdown(
    f"""
    <style>
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background: rgba(50, 50, 50, 100)
        }}
    </style>
    <div class="footer">
        <p><strong>📊 Model Used:</strong> {model_name}  |   <strong> 🔍 Accuracy:</strong> { model_accuracy}%</p>
    </div>
    """,
    unsafe_allow_html=True,
)
