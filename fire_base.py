import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime  # Import datetime for getting current time
import os

# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate('serviceAccountKey.json')  # Path to your service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv('DATABASE_URL')
})

def fetch_context(uid):
    try:
        return db.reference(f'Users/{uid}/context').get()
    except Exception as e:
        raise Exception(f"Context Fetch Error: {str(e)}")

def put_context(uid, query,  number):
    try:
        context = fetch_context(uid) or []
        
        # Get the current time and date
        timestamp = datetime.now().isoformat()  # You can format this as needed
        
        # Append the new context with timestamp
        context.append({
            "name": query,
            
            "number": number,
            "timestamp": timestamp  # Add timestamp here
        })
        db.reference(f'Users/{uid}/context').set(context)
    except Exception as e:
        raise Exception(f"Context Storage Error: {str(e)}")

# Example usage
