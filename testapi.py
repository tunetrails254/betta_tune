import requests

# Endpoint
url = "http://127.0.0.1:5000/predict"

# Your API key
api_key = "e09aeb6894f3afb0e82f78ba900ce977"


# File path
audio_path = r"C:\Users\HP\Desktop\Career_stuff\betta_tune\uploads\male_fifties_06312.wav"

# Headers
headers = {
    "X-API-KEY": api_key
}

# File to upload
files = {
    "audio": open(audio_path, "rb")
}

# Make POST request
response = requests.post(url, headers=headers, files=files)

# Print result
try:
    print(response.json())
except Exception as e:
    print("Error parsing response:", e)
    print(response.text)
