import google.generativeai as genai

API_KEY = "AQ.Ab8RN6LNaTs9JVIupqA3TXVhf9_ApYykw-s_ibSjhB2BIMcBWw"

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "Who developed this chatbot?"
)

print(response.text)