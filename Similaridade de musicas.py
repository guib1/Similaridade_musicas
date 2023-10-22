import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_input = pd.read_csv('input.csv')

df_input['Letra'] = df_input['Letra'].fillna('')
df_input['Gênero Musical'] = df_input['Gênero Musical'].fillna('')

df_output = pd.read_csv('output.csv')

df_output['features'] = df_output['Nome do Artista'] + ' ' + df_output['Nome da Música'] + ' ' + df_output['Letra'] + ' ' + df_output['Gênero Musical']

vectorizer = TfidfVectorizer()
tfidf_matrix_output = vectorizer.fit_transform(df_output['features'])

def calculate_similarity(input_features, output_features):
    tfidf_matrix_input = vectorizer.transform([input_features])
    cosine_sim = cosine_similarity(tfidf_matrix_input, tfidf_matrix_output)
    return cosine_sim

def get_recommendations_by_input(title, similarity_threshold=0.2):
    if title not in df_input['Nome da Música'].values:
        return "Música não encontrada no DataFrame de entrada."

    idx_input = df_input[df_input['Nome da Música'] == title].index[0]
    input_features = df_input.loc[idx_input, 'Nome do Artista'] + ' ' + df_input.loc[idx_input, 'Nome da Música'] + ' ' + df_input.loc[idx_input, 'Letra'] + ' ' + df_input.loc[idx_input, 'Gênero Musical']

    similarity = calculate_similarity(input_features, df_output['features'])
    
    sim_scores = list(enumerate(similarity[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommended_songs = [df_output.loc[i, 'Nome da Música'] for i, score in sim_scores if score >= similarity_threshold]
    
    return recommended_songs

for index, row in df_input.iterrows():
    input_title = row['Nome da Música']
    recommended_songs = get_recommendations_by_input(input_title)
    
    input_artist = df_output['Nome do Artista']
    
    print(f"Recomendações para '{input_title}':")
    if recommended_songs:
        print(recommended_songs)
    else:
        print("Nenhuma recomendação encontrada com base no limite de similaridade.")
