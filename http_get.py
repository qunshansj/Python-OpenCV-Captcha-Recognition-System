python
import requests

def http_get(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

if __name__ == "__main__":
    url = "https://example.com"
    data = http_get(url)
    if data:
        print(data)
    else:
        print("Failed to get data from", url)
