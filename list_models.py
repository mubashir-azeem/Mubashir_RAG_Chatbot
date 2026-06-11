import google.generativeai as genai

genai.configure(api_key="AQ.Ab8RN6LNaTs9JVIupqA3TXVhf9_ApYykw-s_ibSjhB2BIMcBWw")

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)