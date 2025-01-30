import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_PATH = Path("D:/My Project/Project Intro to DS/Project DS/Disease_Prediction/Dataset")
RANDOM_STATE = 44


symptom_translations = {
    'itching': 'រមាស់',
    'skin_rash': 'រលាកស្បែក',
    'nodal_skin_eruptions': 'រោគសើរស្បែក',
    'continuous_sneezing': 'កណ្តស់ជាប់ៗគ្នា',
    'shivering': 'ញ័រញ៉ាក់',
    'chills': 'រងារ',
    'joint_pain': 'ឈឺសន្លាក់',
    'stomach_pain': 'ឈឺពោះ',
    'acidity': 'ឡើងជាត់អាស៊ីត',
    'ulcers_on_tongue': 'ស្នាមជាំលើអណ្តាត',
    'muscle_wasting': 'របួសដាច់សាច់ដុំ',
    'vomiting': 'ហើម',
    'burning_micturition': 'ឈឺនៅពេលបញ្ចេញទឹកនោម',
    'spotting_urination': 'នោមញឹក',
    'fatigue': 'អស់កំលាំង',
    'weight_gain': 'ឡើងទម្ងន់',
    'anxiety': 'ខ្វល់ខ្វាយច្រើន',
    'cold_hands_and_feets': 'ដៃនិងជើងត្រជាក់',
    'mood_swings': 'ប្តូរអារម្មណ៍គិតច្រើន',
    'weight_loss': 'ស្រកគីឡួ',
    'restlessness': 'ភាពមិនសុខស្រួលក្នុងខ្លួន',
    'lethargy': 'ភាពអស់កំលាំង',
    'patches_in_throat': 'បន្ទាត់ក្នុងបំពង់ក',
    'irregular_sugar_level': 'កម្រិតស្ករមិនទៀងទាត់',
    'cough': 'ក្អក',
    'high_fever': 'គ្រុនខ្លាំង',
    'sunken_eyes': 'ភ្នែកចុះច្រាស',
    'breathlessness': 'ភាពមិនអាចដកដង្ហើមបាន',
    'sweating': 'ការបែកញើស',
    'dehydration': 'ភាពស្រេកទឹក',
    'indigestion': 'ការមិនទទួលទានអាហារបានល្អ',
    'headache': 'ឈឺក្បាល',
    'yellowish_skin': 'ស្បែកលឿង',
    'dark_urine': 'ទឹកនោមងងឹត',
    'nausea': 'ការមិនស្រួលក្នុងពោះ',
    'loss_of_appetite': 'បាត់បង់ចំណង់អាហារ',
    'pain_behind_the_eyes': 'ឈឺពីក្រោយភ្នែក',
    'back_pain': 'ឈឺខ្នង',
    'constipation': 'រំខានពោះវៀន',
    'abdominal_pain': 'ឈឺក្នុងក្រលៀន',
    'diarrhoea': 'រាក',
    'mild_fever': 'គ្រុនស្រាល',
    'yellow_urine': 'ទឹកនោមលឿង',
    'yellowing_of_eyes': 'ភ្នែកលឿង',
    'acute_liver_failure': 'ការបរាជ័យក្រពះ',
    'fluid_overload': 'ការកាត់បន្ថយហើយគ្មានអ្វីកើតឡើង',
    'swelling_of_stomach': 'ការហើមពោះ',
    'swelled_lymph_nodes': 'ការហើមកញ្ចឹងទឹក',
    'malaise': 'សេចក្តីលំបាក',
    'blurred_and_distorted_vision': 'ភាពភ្លឺខុសគ្នានិងខូចខាត',
    'phlegm': 'ការជិតឬគយ',
    'throat_irritation': 'ការលំបាកបំពង់ក',
    'redness_of_eyes': 'ភ្នែកក្រហម',
    'sinus_pressure': 'សម្ពាធនៃបំពង់មុខ',
    'runny_nose': 'ច្រមុះហូរទឹក',
    'congestion': 'ការរមៀត',
    'chest_pain': 'ឈឺទ្រូង',
    'weakness_in_limbs': 'ភាពទន់ខ្សោយក្នុងសាច់ដុំ',
    'fast_heart_rate': 'អត្រាបេះដូងលឿន',
    'pain_during_bowel_movements': 'ឈឺនៅពេលបន្តក់រោម',
    'pain_in_anal_region': 'ឈឺនៅក្នុងត្រឡាច',
    'bloody_stool': 'ចំណីឈាម',
    'irritation_in_anus': 'ការលំបាកក្នុងត្រឡាច',
    'neck_pain': 'ឈឺក',
    'dizziness': 'ឧទ្ទិស',
    'cramps': 'អន្ទាក់',
    'bruising': 'ស្នាមជាំ',
    'obesity': 'ជំងឺអាហារលើស',
    'swollen_legs': 'ជើងហើម',
    'swollen_blood_vessels': 'សរសៃឈាមហើម',
    'puffy_face_and_eyes': 'មុខនិងភ្នែកហើម',
    'enlarged_thyroid': 'ក្រពះធំ',
    'brittle_nails': 'ក្រចកខូច',
    'swollen_extremities': 'សាច់ដុំខ្សោយ',
    'excessive_hunger': 'ឃ្លាន',
    'extra_marital_contacts': 'ទំនាក់ទំនងក្រៅពីអាពាហ៍ពិពាហ៍',
    'drying_and_tingling_lips': 'សន្សើមនិងឈឺមាត់',
    'slurred_speech': 'ការនិយាយលឿន',
    'knee_pain': 'ឈឺជង្គង់',
    'hip_joint_pain': 'ឈឺសន្លាក់',
    'muscle_weakness': 'ភាពទន់ខ្សោយសាច់ដុំ',
    'stiff_neck': 'ករឹង',
    'swelling_joints': 'សន្លាក់ហើម',
    'movement_stiffness': 'ភាពរឹងក្នុងចលនា',
    'spinning_movements': 'ចលនាទ្រេតទ្រោធ',
    'loss_of_balance': 'បាត់សមទិន',
    'unsteadiness': 'មិនទៀងទាត់',
    'weakness_of_one_body_side': 'ទន់ខ្សោយមួយចំហៀងខ្លួន',
    'loss_of_smell': 'បាត់ចរិតក្លិន',
    'bladder_discomfort': 'ការមិនស្រួលក្នុងខ្នង',
    'foul_smell_of_urine': 'ទឹកនោមមានក្លិនស្អុយ',
    'continuous_feel_of_urine': 'មានអារម្មណ៍ចង់តែនោម',
    'passage_of_gases': 'ផោមញឹក',
    'internal_itching': 'ឈឺសាច់ដុំផ្នែកក្នុង',
    'toxic_look_(typhos)': 'ស្មុគស្មាញ',
    'depression': 'ផ្លូវចិត្តមិនហ្នឹងនរ',
    'irritability': 'ភាពរំខាន',
    'muscle_pain': 'ឈឺសាច់ដុំ',
    'altered_sensorium': 'ភាពផ្លាស់ប្តូរនៃចរិត',
    'red_spots_over_body': 'កន្ទួលក្រហមលើខ្លួន',
    'belly_pain': 'ឈឺពោះ',
    'abnormal_menstruation': 'ប្រការជំងឺចុងសម្រាម',
    'dischromic_patches': 'ស្នាមបន្ទាត់លើខ្លួន',
    'watering_from_eyes': 'ទឹកហូរពីភ្នែក',
    'increased_appetite': 'ពោះវៀនបន្ថែម',
    'polyuria': 'ការបញ្ចេញទឹកនោមច្រើន',
    'family_history': 'ប្រវត្តិសាច់ញាតិ',
    'mucoid_sputum': 'ទឹកនោមមានភ្លឺ',
    'rusty_sputum': 'ទឹកភ្លឺចាស់',
    'lack_of_concentration': 'បាត់បង់ការយល់ដឹង',
    'visual_disturbances': 'ការលំបាកក្នុងការមើល',
    'receiving_blood_transfusion': 'ការទទួលការបញ្ចូលឈាម',
    'receiving_unsterile_injections': 'ការទទួលការចាក់វ៉ាក់ស៊ីនមិនស្អាត',
    'coma': 'ជំងឺដេកមិនដឹងខ្លួន',
    'stomach_bleeding': 'ឈាមហូរពោះ',
    'distention_of_abdomen': 'ការហើមនៃផ្ទៃពោះ',
    'history_of_alcohol_consumption': 'ធ្លាប់ផឹកស្រា',
    'fluid_overload': 'ឡើងគីឡូ',
    'blood_in_sputum': 'ឈាមក្នុងទឹកនោម',
    'prominent_veins_on_calf': 'សរសៃឈាមឡើងធំនៅជើង',
    'palpitations': 'បេះដូងលោតលឿន',
    'painful_walking': 'ឈឺក្នុងការដើរ',
    'pus_filled_pimples': 'ស្បែកមុខមានពងបែក',
    'blackheads': 'មុនខ្មៅ',
    'scurring': 'ស្ពឹកស្បែក',
    'skin_peeling': 'ស្បែករបក',
    'silver_like_dusting': 'ក្អែលក',
    'small_dents_in_nails': 'ស្នាមជាំតូចលើក្រចក',
    'inflammatory_nails': 'រលាកក្រចក',
    'blister': 'ពងមុន',
    'red_sore_around_nose': 'ស្នាមជាំក្រហមជុំវិញច្រមុះ',
    'yellow_crust_ooze': 'ស្នាមរបួសលឿង'
}

@st.cache_data
def load_data():
    """Load datasets from the specified path."""
    try:
        df = pd.read_csv(DATA_PATH / "dataset.csv")
        ds = pd.read_csv(DATA_PATH / "symptom_Description.csv")
        pr = pd.read_csv(DATA_PATH / "symptom_precaution.csv")
        return df, ds, pr
    except FileNotFoundError as e:
        st.error("Error loading data. Ensure file paths are correct.")
        logger.error(f"FileNotFoundError: {str(e)}")
        return None, None, None

@st.cache_data
def preprocess_data(df):
    """Preprocess the symptoms dataset."""
    if df is None:
        return None
    df = df.copy()
    df['Symptom'] = df[[f'Symptom_{i}' for i in range(1, 18)]].fillna('').agg(' '.join, axis=1)
    return df[['Symptom', 'Disease']]

@st.cache_data
def prepare_precautions(pr):
    """Prepare the precautions dataset."""
    if pr is None:
        return None
    pr = pr.copy()
    pr['precautions'] = pr[[f'Precaution_{i}' for i in range(1, 5)]].fillna('').agg(', '.join, axis=1)
    return pr[['Disease', 'precautions']]

@st.cache_resource
def train_model(x_train, y_train):
    """Train the machine learning model."""
    try:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LinearSVC(random_state=RANDOM_STATE)),
        ])
        pipeline.fit(x_train, y_train)
        return pipeline
    except Exception as e:
        st.error("Model training failed. Check the data or configurations.")
        logger.error(f"Training Error: {str(e)}")
        return None

def display_prediction_results(model, symptoms, ds, pr):
    """Display prediction results along with description and precautions."""
    try:
        disease_predicted = model.predict([symptoms])[0]
        description = ds.loc[ds['Disease'] == disease_predicted, 'Description'].iloc[0]
        precautions = pr.loc[pr['Disease'] == disease_predicted, 'precautions'].iloc[0] \
            if disease_predicted in pr['Disease'].values else 'Precautions not available.'

        st.subheader("Prediction Result")
        st.write(f"**Disease Predicted:** {disease_predicted}")
        st.write(f"**Description:** {description}")
        st.write(f"**Precautions:** {precautions}")
    except Exception as e:
        st.error("An error occurred while displaying the prediction results.")
        logger.error(f"Result Display Error: {str(e)}")

def main():
    st.title("Disease Prediction Using Machine Learning")
    st.markdown(
        """This app predicts diseases based on symptoms using a machine learning model.
        Provide your symptoms and receive potential disease predictions along with 
        descriptions and precautionary measures."""
    )

    df, ds, pr = load_data()
    if df is None or ds is None or pr is None:
        return


    df = preprocess_data(df)
    pr = prepare_precautions(pr)


    x = df['Symptom']
    y = df['Disease']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, shuffle=True
    )


    model = train_model(x_train, y_train)
    if model is None:
        return

    st.sidebar.title("Input Symptoms")
    input_method = st.sidebar.radio("Select Input Method:", ["Quick Choice", "Typing"])

    symptom_list = [
        "abdominal_pain", "abnormal_menstruation", "acidity", "acute_liver_failure", 
        "altered_sensorium", "anxiety", "back_pain", "belly_pain", "blackheads", 
        "bladder_discomfort", "blister", "blood_in_sputum", "bloody_stool", 
        "blurred_and_distorted_vision", "breathlessness", "brittle_nails", 
        "bruising", "burning_micturition", "chest_pain", "chills", 
        "cold_hands_and_feets", "coma", "congestion", "constipation", 
        "continuous_feel_of_urine", "continuous_sneezing", "cough", "cramps", 
        "dark_urine", "dehydration", "depression", "diarrhoea", 
        "dischromic_patches", "distention_of_abdomen", "dizziness", 
        "drying_and_tingling_lips", "enlarged_thyroid", "excessive_hunger", 
        "extra_marital_contacts", "family_history", "fast_heart_rate", "fatigue", 
        "fluid_overload", "foul_smell_of_urine", "headache", "high_fever", 
        "hip_joint_pain", "history_of_alcohol_consumption", "increased_appetite", 
        "indigestion", "inflammatory_nails", "internal_itching", 
        "irregular_sugar_level", "irritability", "irritation_in_anus", 
        "joint_pain", "knee_pain", "lack_of_concentration", "lethargy", 
        "loss_of_appetite", "loss_of_balance", "loss_of_smell", "malaise", 
        "mild_fever", "mood_swings", "movement_stiffness", "mucoid_sputum", 
        "muscle_pain", "muscle_wasting", "muscle_weakness", "nausea", 
        "neck_pain", "nodal_skin_eruptions", "obesity", "pain_behind_the_eyes", 
        "pain_during_bowel_movements", "pain_in_anal_region", "painful_walking", 
        "palpitations", "passage_of_gases", "patches_in_throat", "phlegm", 
        "polyuria", "prominent_veins_on_calf", "puffy_face_and_eyes", 
        "pus_filled_pimples", "receiving_blood_transfusion", 
        "receiving_unsterile_injections", "red_sore_around_nose", 
        "red_spots_over_body", "redness_of_eyes", "restlessness", "runny_nose", 
        "rusty_sputum", "scurring", "shivering", "silver_like_dusting", 
        "sinus_pressure", "skin_peeling", "skin_rash", "slurred_speech", 
        "small_dents_in_nails", "spinning_movements", "spotting_urination", 
        "stiff_neck", "stomach_bleeding", "stomach_pain", "sunken_eyes", 
        "sweating", "swelled_lymph_nodes", "swelling_joints", 
        "swelling_of_stomach", "swollen_blood_vessels", "swollen_extremeties", 
        "swollen_legs", "throat_irritation", "toxic_look_(typhos)", 
        "ulcers_on_tongue", "unsteadiness", "visual_disturbances", "vomiting", 
        "watering_from_eyes", "weakness_in_limbs", "weakness_of_one_body_side", 
        "weight_gain", "weight_loss", "yellow_crust_ooze", "yellow_urine", 
        "yellowing_of_eyes", "yellowish_skin", "itching"
    ]

    if input_method == "Quick Choice":
        selected_symptoms = st.sidebar.multiselect(
            "Select symptoms:",
            options=symptom_list,
            format_func=lambda x: f"{x} ({symptom_translations.get(x, 'No translation')})"
        )
        symptoms = " ".join(selected_symptoms)
    else:
        symptoms = st.sidebar.text_area("Enter symptoms (space-separated):", help="Example: itching skin_rash cough")

    if symptoms:
        display_prediction_results(model, symptoms, ds, pr)

    if st.sidebar.checkbox("Show Model Performance"):
        predictions = model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predictions)

        st.subheader("Model Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")

        with col2:
            st.metric("Diseases Covered", len(df['Disease'].unique()))

        st.write("**Classification Report:**")
        st.code(metrics.classification_report(y_test, predictions))

if __name__ == "__main__":
    main()