from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
from newtest import main_bot
from twilio.rest import Client
import time
import requests
from requests.auth import HTTPBasicAuth
from faster_whisper import WhisperModel
import os
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from firebase import firebase_admin
from firebase import put_context
# Path to the service account key JSON file
from dsecription import get_disease_description

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
app = Flask(__name__)

# Twilio credentials (replace with your actual credentials)
TWILIO_ACCOUNT_SID = os.getenv('SSID')
TWILIO_AUTH_TOKEN = os.getenv('AUTH_TOKEN')
TWILIO_PHONE_NUMBER = '+16103645557'

# Initialize the Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Placeholder for your disease prediction model
def predict_disease(x):
    url = 'https://disease-prediction-u62r.onrender.com/create'  # Replace with your API endpoint
    data = {
        "query": x
    }

    # Send the POST request
    
    response= requests.post(url, json=data)
    print(response)
    disease_prediction=response.json()["result"]
    res=get_disease_description(disease_prediction)
    disease_res=res

    return disease_prediction,disease_res

# Google Speech-to-Text transcription function
# Route to handle incoming calls and record audio
@app.route("/ivr", methods=['POST'])
def ivr():
    response = VoiceResponse()
    
    # Record speech input instead of using Twilio's gather for speech recognition
    response.say("welcome to Heal O")
    response.say("Please describe your symptoms after the beep.")
    response.record(timeout=2.5, action='/process_recording', transcribe=False)
    
    return str(response)

# Route to process the recorded audio
@app.route("/process_recording", methods=['POST'])
def process_recording():
    recording_url = request.form.get('RecordingUrl')  # Get the recording URL
    print(f"Recording URL: {recording_url}")
    
    # Transcribe the recording using Google Speech-to-Text
    recording_url = request.form.get('RecordingUrl')  # Get the recording URL
    print(f"Recording URL: {recording_url}")
    recording_url=recording_url.strip()
    # Send GET request to the URL using basic auth
    time.sleep(3)
    response = requests.get(recording_url, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),verify=False)

# Check if the request was successful
   
# Check if the request was successful
    if response.status_code == 200:
        # Save the audio file locally
        audio_file = "twilio_recording.mp3"
        with open(audio_file, "wb") as file:
            file.write(response.content)
        print("Audio file downloaded and saved as 'twilio_recording.mp3'")
    else:
        print(f"Failed to download audio. Status code: {response.status_code}")
        exit()

    # Load the tiny Whisper model
    model = WhisperModel("tiny", device="cpu")
    segments, info = model.transcribe(audio_file)
    x=""

    for segment in segments:
        x+= segment.text
    print(x)

    # Transcribe the audio file
    
    response = VoiceResponse()
    global disease_prediction,disease_res
    # Perform disease prediction based on transcription
    disease_prediction, disease_res = predict_disease(x)
    
    if disease_prediction:
        # Respond based on the disease prediction
        response.say(disease_prediction)
        response.say(disease_res)
        response.say("Would you like to fix an appointment with our finest doctor? If yes, press 1. Otherwise, press any other number.")
        response.gather(input='dtmf', timeout=4, num_digits=1, action='/digit_gather')
        


    else:
        # If no specific disease could be determined, restart the IVR
        response.say("I'm sorry, I couldn't determine any specific disease. Let's try again.")
        response.redirect('/ivr')  # Redirect back to the IVR function
    
    return str(response)

# Route to handle DTMF input for appointment booking
@app.route("/digit_gather", methods=['POST'])
def digit_gather():
    digits = request.form.get('Digits')  # Correct way to get the DTMF input
    response = VoiceResponse()
    time.sleep(2)
    
    if digits == '1':
        response.say("You pressed 1 to fix an appointment.")
        
        # Send SMS confirmation
        response.say("give us your nick name")
        response.record(timeout=5, action='/process_name_recording', transcribe=False)
    else:
        response.say("You have not opted for appointment booking.")
        response.say("Thank You")
    
    return str(response)

# Root route for testing

@app.route("/process_name_recording", methods=['POST'])
def process_name_recording():
    recording_url = request.form.get('RecordingUrl') 
    response=VoiceResponse() # Get the recording URL for the name
    print(f"Name Recording URL: {recording_url}")
    time.sleep(2)
    # Download the name recording
    response = requests.get(recording_url, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), verify=False)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the audio file locally
        audio_file = "name_recording.mp3"
        with open(audio_file, "wb") as file:
            file.write(response.content)
        print("Name audio file downloaded and saved as 'name_recording.mp3'")

        # Load the tiny Whisper model for name transcription
        model = WhisperModel("tiny", device="cpu")
        segments, info = model.transcribe(audio_file)
        name_transcription = ""

        for segment in segments:
            name_transcription += segment.text
        print(f"Recorded Name: {name_transcription}")
        message = client.messages.create(
            body=f"Thank you for choosing to book an appointment. {name_transcription} have been diagonised with {disease_prediction}",
            from_=TWILIO_PHONE_NUMBER,
            to=request.values.get('From')  # Send SMS to the caller's number
        )

        # Store the user's name into Firebase or perform any required action
        # Assuming you have a variable `uid` that represents the user ID
        uid = "u2"  # Replace with the actual user ID as necessary
        number=str(request.values.get('From'))
        put_context(uid, name_transcription,number)  # You can modify the parameters if needed

        
    else:
        print(f"Failed to download name audio. Status code: {response.status_code}")
    
    return str(response)
@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run(debug=True)
