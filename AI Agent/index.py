import requests
from google import genai
from os import getenv
from dotenv import load_dotenv

load_dotenv()

api_key = getenv("API_KEY")
weather_api_key = getenv("WEATHER_API_KEY")
model = getenv("MODEL")


def sum_of_two_number(numbers):
    return numbers["num1"] + numbers["num2"]


def check_prime_number(numbers):
    num = numbers["num"]
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True


def getWeather(weather):
    city = weather["city"]
    response = requests.get(
        f"https://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}"
    )
    data = response.json()
    return f"Temprature: {data['current']['temp_c']} °C\nCondition: {data['current']['condition']['text']}\nWind speed: {data['current']['wind_kph']} kmph\nHumidity: {data['current']['humidity']} %"


weather_tool = {
    "type": "function",
    "name": "getWeather",
    "description": """This will accept the city name as parameter and give you the Temperature in degree celcius , Condition of weather of that city that can be a sunny , Wind speed in kmph and humidity in percentage. 
    e.g:- Temprature: 29.4 °C
          Condition: Smoky haze
          Wind speed: 14.8 kmph
          Humidity: 59 % """,
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. Vasai, Mumbai, Delhi",
            },
        },
        "required": ["city"],
    },
}

sum2number_tool = {
    "type": "function",
    "name": "sum_of_two_number",
    "description": "This function accept two numbers as parameter and return the sum of that two numbers. eg:- {10 , 20}",
    "parameters": {
        "type": "object",
        "properties": {
            "num1": {
                "type": "number",
                "description": "The first number",
            },
            "num2": {
                "type": "number",
                "description": "The second number",
            },
        },
        "required": ["num1", "num2"],
    },
}

check_prime_number_tool = {
    "type": "function",
    "name": "check_prime_number",
    "description": "This function tells us that the given number is prime number or not. It will take only one number as a parameter and then gives us result",
    "parameters": {
        "type": "object",
        "properties": {
            "num": {
                "type": "number",
                "description": "This is the number which is given as a parameter to check that it is prime number or not. eg:- {10}",
            },
        },
        "required": ["num"],
    },
}

history = []
client = genai.Client(api_key=api_key)

availableTool = {
    "sum_of_two_number": sum_of_two_number,
    "check_prime_number": check_prime_number,
    "getWeather": getWeather,
}

tools = [weather_tool, sum2number_tool, check_prime_number_tool]

def create_interaction():
    return client.interactions.create(
                model=model,
                input=history,
                system_instruction="You are an intelligent AI. Always provide the final answer as plain text, without Markdown.",
                tools=tools,
                generation_config={
                    "temperature" : 0,
                },
                store=False,
            )


def main():
    try:
        while True:
            userProblem = input("User: ")

            if userProblem.lower() == "exit":
                break

            history.append(
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": userProblem}],
                }
            )

            interaction = create_interaction()

            while True:
                function_call = [
                    step
                    for step in (interaction.steps or [])
                    if step.type == "function_call"
                ]

                if not function_call:
                    break

                for step in interaction.steps:
                    history.append(step.model_dump())

                for step in function_call:
                    result = availableTool[step.name](step.arguments)

                    history.append(
                        {
                            "type": "function_result",
                            "name": step.name,
                            "call_id": step.id,
                            "result": [{"type": "text", "text": str(result)}],
                        }
                    )

                interaction = create_interaction()

            history.append(
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": interaction.output_text}],
                }
            )

            print("AI: ", interaction.output_text)

    except Exception as e:
        print(f"Error: {e}")


main()