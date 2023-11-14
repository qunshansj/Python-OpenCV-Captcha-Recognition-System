python
import requests
import json

class WeatherAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_weather(self, city):
        url = f"http://api.weatherapi.com/v1/current.json?key={self.api_key}&q={city}"
        response = requests.get(url)
        data = json.loads(response.text)
        temperature = data['current']['temp_c']
        condition = data['current']['condition']['text']
        return f"The current temperature in {city} is {temperature}°C with {condition}."

api_key = "YOUR_API_KEY"
weather_api = WeatherAPI(api_key)
city = input("Enter city name: ")
print(weather_api.get_weather(city))
