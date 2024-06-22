import streamlit as st
import openai
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Configurer votre clé OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Chargement du modèle de vectorisation
model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

# Fonction pour récupérer les données de la base de données
def get_data_from_db(db_path='pdf_data.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, text, vector FROM pdf_content')
    rows = c.fetchall()
    conn.close()
    return rows

# Fonction pour décoder le vecteur
def decode_vector(vector_blob):
    return np.frombuffer(vector_blob, dtype=np.float32)

# Fonction principale pour interagir avec l'utilisateur et récupérer les données appropriées
def chatbot_main(question):
    # Vectorisation de la question de l'utilisateur
    question_vector = model.encode(question, convert_to_tensor=True)

    # Connexion à la base de données et récupération des données
    rows = get_data_from_db()

    best_match = None
    best_similarity = -1

    # Recherche de la meilleure réponse
    for row_id, text, vector_blob in rows:
        vector = decode_vector(vector_blob)
        similarity = float(torch.cosine_similarity(question_vector, torch.tensor(vector), dim=0))
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = (row_id, text)

    return best_match

# Fonction pour obtenir le reste des étapes en utilisant l'API OpenAI
def get_next_step(text_with_question):
    prompt = f"Le texte suivant décrit les étapes d'un processus ou un mode opératoire, merci de ne pas inventer des réponses, et de répondre à la demande:\n\n{text_with_question}"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=800,
        n=1,
        stop=None,
        temperature=0.1
    )
    return response.choices[0].text.strip()

# Interface Streamlit
def main():
    st.title("Chatbot SALAFIN")

    user_input = st.text_input("Posez votre question:")

    if user_input:
        # Obtenir la réponse appropriée
        match = chatbot_main(user_input)

        if match:
            row_id, text = match
            text_with_question = f"{text}\n\nQuestion utilisateur : {user_input}"
            next_step = get_next_step(text_with_question)
            if next_step:
                st.write("### Réponse :")
                st.write(next_step)
            else:
                st.write("### Réponse :")
                st.write("Désolé, je n'ai pas pu trouver le reste des étapes.")
        else:
            st.write("### Réponse :")
            st.write("Désolé, je n'ai pas pu trouver de réponse à votre question.")

if __name__ == "__main__":
    main()
