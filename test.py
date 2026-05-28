import requests

url = "http://127.0.0.1:5000/predict"

image_path = r"C:\Users\Admin\OneDrive\Desktop\CNN\PlantVillageDataset\Tomato_Early_blight\ff83852f-65e0-4981-8c66-6b86d34b32c2___RS_Erly.B 9582.JPG"

try:
    with open(image_path, 'rb') as f:
        response = requests.post(url, files={'image': f})
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")

except Exception as e:
    print(f"Error: {e}")