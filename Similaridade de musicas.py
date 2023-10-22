import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar dados do arquivo CSV de entrada (input.csv)
df_input = pd.read_csv('input.csv')  # Substitua 'input.csv' pelo caminho do seu arquivo de entrada CSV

# Preencher valores NaN nas colunas 'Letra' e 'Gênero Musical' com strings vazias
df_input['Letra'] = df_input['Letra'].fillna('')
df_input['Gênero Musical'] = df_input['Gênero Musical'].fillna('')

# Carregar dados do arquivo CSV de saída (output.csv)
df_output = pd.read_csv('output.csv')  # Substitua 'output.csv' pelo caminho do seu arquivo de saída CSV

# Combinar todas as características em uma única string para o arquivo de saída
df_output['features'] = df_output['Nome do Artista'] + ' ' + df_output['Nome da Música'] + ' ' + df_output['Letra'] + ' ' + df_output['Gênero Musical']

# Calcular a matriz TF-IDF para as músicas de saída
vectorizer = TfidfVectorizer()
tfidf_matrix_output = vectorizer.fit_transform(df_output['features'])

# Função para calcular a similaridade com base na letra e no gênero
def calculate_similarity(input_features, output_features):
    tfidf_matrix_input = vectorizer.transform([input_features])
    cosine_sim = cosine_similarity(tfidf_matrix_input, tfidf_matrix_output)
    return cosine_sim

# Função para obter recomendações com base nas músicas de entrada
def get_recommendations_by_input(title, similarity_threshold=0.2):
    # Verificar se o título da música existe no DataFrame de entrada
    if title not in df_input['Nome da Música'].values:
        return "Música não encontrada no DataFrame de entrada."

    # Calcular a matriz TF-IDF para a música de entrada
    idx_input = df_input[df_input['Nome da Música'] == title].index[0]
    input_features = df_input.loc[idx_input, 'Nome do Artista'] + ' ' + df_input.loc[idx_input, 'Nome da Música'] + ' ' + df_input.loc[idx_input, 'Letra'] + ' ' + df_input.loc[idx_input, 'Gênero Musical']

    # Calcular a matriz de similaridade de cosseno entre a música de entrada e as músicas de saída
    similarity = calculate_similarity(input_features, df_output['features'])
    
    # Obter as pontuações de similaridade
    sim_scores = list(enumerate(similarity[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obter as músicas recomendadas com base na letra e no gênero, considerando o similarity_threshold
    recommended_songs = [df_output.loc[i, 'Nome da Música'] for i, score in sim_scores if score >= similarity_threshold]
    
    return recommended_songs

# Loop para obter recomendações para cada música de entrada
for index, row in df_input.iterrows():
    input_title = row['Nome da Música']
    recommended_songs = get_recommendations_by_input(input_title)
    
    input_artist = df_output['Nome do Artista']
    
    print(f"Recomendações para '{input_title}':")
    if recommended_songs:
        print(recommended_songs)
    else:
        print("Nenhuma recomendação encontrada com base no limite de similaridade.")