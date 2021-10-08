import requests

url = 'http://localhost:3000/predict_api'
r = requests.post(url,json={'Age':23, 'Sex':2, 'BP':1, 
                            'Cholesterol':2, 'Na_to_K':25.234})

print(r.json())