import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':28, 'wage':1.0, 'yoc':15.0 , 'ca':78.0, 'pa':78.0})

print(r.json())