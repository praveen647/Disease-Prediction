import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time
import os
from fastapi import FastAPI, HTTPException
import uvicorn
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import pyrebase
import re
import spacy
from difflib import get_close_matches
import joblib
import smtplib
import random
import string
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import requests
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
sym_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
nlp = spacy.load("en_core_web_sm")
loaded_encoder = joblib.load('label_encoder.pkl')
API_KEY = os.getenv('API_KEY')
symptom_dict = sym_dict
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API')
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3,max_tokens=None)
system_prompt = (
    "You are an assistant for question-answering tasks about medical data and it's remedies "
    "Use the following context to answer"
    "the question.Highlight the problem in bold. If you don't know the answer, say that you "
    "don't know and don't mention that you are provided with context and act like human. Generate your reply to be conversational"
    "don't stick only to the context mention things you know"
    "\n\n"
    "{context}"
)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

def preprocess(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^a-z\s]', '', user_input)
    doc = nlp(user_input)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens
def get_token():
  token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
  API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
  mltoken = token_response.json()["access_token"]
  return mltoken
def call_model(arr):
  arr = arr.tolist()
  mltoken = get_token()
  header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
  payload_scoring = {"input_data": [{"field":[
    "itching",
    "skin rash",
    "nodal skin eruptions",
    "continuous sneezing",
    "shivering",
    "chills",
    "joint pain",
    "stomach pain",
    "acidity",
    "ulcers on tongue",
    "muscle wasting",
    "vomiting",
    "burning micturition",
    "spotting urination",
    "fatigue",
    "weight gain",
    "anxiety",
    "cold hands and feet",
    "mood swings",
    "weight loss",
    "restlessness",
    "lethargy",
    "patches in throat",
    "irregular sugar level",
    "cough",
    "high fever",
    "sunken eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish skin",
    "dark urine",
    "nausea",
    "loss of appetite",
    "pain behind the eyes",
    "back pain",
    "constipation",
    "abdominal pain",
    "diarrhoea",
    "mild fever",
    "yellow urine",
    "yellowing of eyes",
    "acute liver failure",
    "fluid overload",
    "swelling of stomach",
    "swelled lymph nodes",
    "malaise",
    "blurred and distorted vision",
    "phlegm",
    "throat irritation",
    "redness of eyes",
    "sinus pressure",
    "runny nose",
    "congestion",
    "chest pain",
    "weakness in limbs",
    "fast heart rate",
    "pain during bowel movements",
    "pain in anal region",
    "bloody stool",
    "irritation in anus",
    "neck pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen legs",
    "swollen blood vessels",
    "puffy face and eyes",
    "enlarged thyroid",
    "brittle nails",
    "swollen extremities",
    "excessive hunger",
    "extra marital contacts",
    "drying and tingling lips",
    "slurred speech",
    "knee pain",
    "hip joint pain",
    "muscle weakness",
    "stiff neck",
    "swelling joints",
    "movement stiffness",
    "spinning movements",
    "loss of balance",
    "unsteadiness",
    "weakness of one body side",
    "loss of smell",
    "bladder discomfort",
    "foul smell of urine",
    "continuous feel of urine",
    "passage of gases",
    "internal itching",
    "toxic look (typhos)",
    "depression",
    "irritability",
    "muscle pain",
    "altered sensorium",
    "red spots over body",
    "belly pain",
    "abnormal menstruation",
    "dichromic patches",
    "watering from eyes",
    "increased appetite",
    "polyuria",
    "family history",
    "mucoid sputum",
    "rusty sputum",
    "lack of concentration",
    "visual disturbances",
    "receiving blood transfusion",
    "receiving unsterile injections",
    "coma",
    "stomach bleeding",
    "distention of abdomen",
    "history of alcohol consumption",
    "blood in sputum",
    "prominent veins on calf",
    "palpitations",
    "painful walking",
    "pus filled pimples",
    "blackheads",
    "scurring",
    "skin peeling",
    "silver-like dusting",
    "small dents in nails",
    "inflammatory nails",
    "blister",
    "red sore around nose",
    "yellow crust ooze"], "values":arr}]}
  response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/diseasepred/predictions?version=2021-05-01', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
  output = response_scoring.json()
  return output['predictions'][0]['values'][0]
def predict(input):
    predicted_labels = call_model(np.expand_dims(input,0))
    print(predicted_labels)
    predicted_labels = np.array(predicted_labels)
    if(np.max(predicted_labels) < 0.5) :
        return "not found in our Database"
    predicted_classes = predicted_labels.argmax(axis=-1)
    print(predicted_classes)
    predicted_classes = [predicted_classes]
    original_labels = loaded_encoder.inverse_transform(np.array(predicted_classes))
    print(original_labels)
    return original_labels
def map_symptoms_to_onehot(user_symptoms, symptom_dict):
    one_hot_vector = [0] * len(symptom_dict)  # Initialize a zero vector
    
    for symptom in user_symptoms:
        # Fuzzy matching to map user input to known symptoms
        match = get_close_matches(symptom, symptom_dict.keys(), n=1, cutoff=0.5)  # Adjust cutoff as needed
        if match:
            index = symptom_dict[match[0]]
            one_hot_vector[index] = 1
        else:
            print(f"Symptom '{symptom}' not recognized, skipping.")
    
    return one_hot_vector
def send_via_email(name, email, phone, date, location, description):

    sender_email = "healo.healthcare@gmail.com"
    sender_password = os.getenv('PASSWORD')
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = email
    message['Subject'] = "Appointment Successful"
    body = f"""
    <html>
    <body>
        <p>Dear {name},</p>
        <p>We are pleased to inform you that your appointment has been confirmed with the following details:</p>
        <ul>
            <li><strong>Appointment Number:</strong>67252577</li>
            <li><strong>Hospital Name:</strong>Vihaa Hospital - Multi-Speciality hospital</li>
            <li><strong>Address:</strong>3rd Ave, Block E, Annanagar East, Chennai, Tamil Nadu 600102</li>
            <li><strong>Appointment Date:</strong>{date}</li>
            <li><strong>Appointment Time:</strong>2 AM</li>
        </ul>
        <p>We look forward to seeing you at the scheduled time. Please arrive at least 15 minutes before your appointment to complete any necessary formalities.</p>
        <p>If you have any questions or need to reschedule, feel free to contact us.</p>
        <p>Best regards,<br>HEAL-O</p>
    </body>
    </html>
    """

    message.attach(MIMEText(body, 'html'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Start TLS (Transport Layer Security)
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()

        print(f"Appointment request confirmation has been sent to {email}")
    except Exception as e:
        print(f"Error: {e}")
config = {
    "apiKey": os.getenv('FIREBASE_API'),
    "authDomain": os.getenv('FIREBASE_AUTH'),
    "databaseURL": os.getenv('DB_URL'),
    "projectId": os.getenv('FIREBASE_ID'),
    "storageBucket": os.getenv('STORAGE_BUCKET'),
    "messagingSenderId": os.getenv('MESSAGING_ID'),
    "appId": os.getenv('APP_ID'),
    "measurementId": os.getenv('MEASUREMENT_ID')
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
WATSON_URL = os.getenv('WATSON_URL')
app=FastAPI()
def ask_watson(prompt):
  body = {
	"input": f"""\"You are an assistant for question-answering tasks about medical data\"
    \"Use the following predicted disease to answer\"
    \"the question.Highlight the problem in bold. If you don'\''t know the answer, say that you \"
    \"don'\''t know and don'\''t mention that you are provided with context and act like human. Generate your reply to be conversational\"
    \"don'\''t stick only to the context mention things you know.\"
    \"\n\n\
    {prompt}
 """,
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 200,
		"repetition_penalty": 1
	},
	"model_id": "ibm/granite-20b-multilingual",
	"project_id": "f566b2c0-0f7a-4e99-be38-f0e22adf64f9",
	"moderations": {
		"hap": {
			"input": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			},
			"output": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			}
		}
	}
}
  token = get_token()
  headers = {
	"Accept": "application/json",
	"Content-Type": "application/json",
	"Authorization": f"Bearer {token}"
  }
  response = requests.post(
	WATSON_URL,
	headers=headers,
	json=body
  )
  if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))
  data = response.json()
  return data['results'][0]['generated_text']
def fetch_context(uid):
    try:
        return db.child("Users").child(uid).child("context").get().val()
    except Exception as e:
        raise Exception(f"Context Fetch Error: {str(e)}")
def generate_prompt(query, context,disease):
    try:
        if context is None:
            prompt = f"""
            Given the predicted disease {disease}
            Answer the following query:
            {query}
            """
        else:
            context_str = " ".join([f"Query: {item['query']} Response: {item['response']} \n" for item in context])
            prompt = f"""
            Given the information of the all the previous conversation below:
            {context_str}
            _________________________________________________________________
            Given the predicted disease {disease}
            Answer the following query in a conversational way in the same language:
            {query}
            """
        return prompt
    except Exception as e:
        raise Exception(f"Prompt Generation Error: {str(e)}")

def put_context(uid, query, response):
    try:
        context = fetch_context(uid)
        if context is None:
            context = []
        context.append({"query": query, "response": response})
        db.child("Users").child(uid).child("context").set(context)
    except Exception as e:
        raise Exception(f"Context Storage Error: {str(e)}")
def put_data(name, email, phone, date, location, description,uid):
    data = {
        "name": name,
        "email": email,
        "phone": phone,
        "date": date,
        "location": location,
        "description": description
    }
    db.child("Users").child(uid).push(data)
class INPUT(BaseModel):
  query:str
  userid:str

class EMAIL(BaseModel):
  name:str
  email:str
  phone:str
  preferredDate:str
  location:str
  description:str
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def base():
  return {"message":"IBM BOT BASE URL"}

@app.post('/create')
def chat(request:INPUT):
  query = request.query
  userid = request.userid
  cleaned_input = preprocess(query)
  cleanest_input = [i for i in cleaned_input if i != ' ']
  if len(cleanest_input)>2:
    one_hot_input = map_symptoms_to_onehot(cleaned_input, symptom_dict)
    print(one_hot_input)
    def is_all_zeroes(lst):
        return sum(lst) == 0
    if(not(is_all_zeroes(one_hot_input))):
        sam =  np.array(one_hot_input,dtype=np.float32)
        disease = predict(one_hot_input)
        print(disease)
        disease = disease[0]
    else:
        disease = "More Symptomps needed"
  else:
    disease = "More Symptomps needed"
  context = fetch_context(userid)
  prompt = generate_prompt(query,context,disease)
  result = chain.invoke({"input":query,"context":prompt})
  put_context(userid,query,result)
  if disease=="More Symptomps needed":
     return {"response":result['text'],"result":None}
  return {"response":result['text'],"result":disease}
@app.post('/send')
def send_mail(request:EMAIL):
  send_via_email(request.name,request.email,request.phone,request.preferredDate,request.location,request.description)
  put_data(request.name,request.email,request.phone,request.preferredDate,request.location,request.description,"U2")

