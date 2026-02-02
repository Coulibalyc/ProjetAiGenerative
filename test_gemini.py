import google.generativeai as genai

API_KEY = "AIzaSyBWA2lb3bcKorg1L3frHkG4e-7qIsrb_3U" # ⚠️ change ta clé, ne jamais la publier !

def test_api():
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content("Hello, are you working?")
        print("✔️ API OK ! Réponse du modèle :")
        print(response.text)
    except Exception as e:
        print("❌ Erreur API :")
        print(e)

test_api()


for m in genai.list_models():
    print(m.name)
