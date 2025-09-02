import requests
url = 'http://localhost:8080/predict'
data = {
    "features":[1,85,66,29,0,26.6,0.351,31]
}
response=requests.post(url,json=data)
print(response.json())