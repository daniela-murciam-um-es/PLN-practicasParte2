import pandas as pd
import json
from datetime import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords

# Intentamos descargar silenciosamente. Si ya están, no hace nada.
try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"⚠️ Advertencia: No se pudieron descargar las stopwords")

## -- PREPROCESADO --

def separar_dataframes(archivo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Separa el archivo json del subreddit que se pase como atributo en dos dataframes de pandas
    uno por submissions y otro por comments.
    '''
    with open(archivo, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extraemos la lista principal
    lista_submissions = data.get("submissions", [])

    # CREAMOS LOS DATAFRAMES
    # Extraemos todos los datos de los hilos, EXCEPTO la lista de comentarios para el de submissions
    # Extraemos todos los comentarios para el de comments
    submissions_limpias = []
    comentarios_planos = []
    for sub in lista_submissions:
        sub_sin_comentarios = {k: v for k, v in sub.items() if k != 'comments'}
        submissions_limpias.append(sub_sin_comentarios)
        for comment in sub.get("comments", []):
            comentarios_planos.append(comment)

    df_submissions = pd.DataFrame(submissions_limpias)
    df_comments = pd.DataFrame(comentarios_planos)

    return df_submissions, df_comments

## -- GRÁFICAS --

def histograma(datos: list[pd.DataFrame], columna: str = 'created_datetime', titulo: str = 'Distribución temporal de', labels: dict = {'created_datetime': 'Fecha y Hora' } ) -> None:
    '''
    Genera histogramas de los dataframes pasados en una lista, sobre el atributo (columna del dataframe) 
    especificado. Distingue si es submission o comment para que quede más claro y se diferencie mejor.
    '''
    subs_dict = {}
    for df in datos:
        sub_name = df['subreddit'].iloc[0]
        if sub_name not in subs_dict:
            subs_dict[sub_name] = {}
        
        # Clasificamos si es submission o comment
        tipo = 'subs' if 'title' in df else 'comms'
        subs_dict[sub_name][tipo] = df

    # Iteramos sobre cada subreddit y creamos su figura de 1x2
    for sub, dfs in subs_dict.items():
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=(f'Submissions (Hilos)', f'Comentarios'),
            horizontal_spacing=0.1
        )

        # Configuración para cada lado
        config = [
            ('subs', 1, '#FF5700', 'Hilos'),
            ('comms', 2, '#336699', 'Comentarios')
        ]

        # Crea el histograma del tipo que sea en ese momento 
        for tipo, col_idx, color, y_label in config:
            if tipo in dfs:
                # Usamos px.histogram para aprovechar su lógica de bins automáticos
                temp_fig = px.histogram(dfs[tipo], x=columna, color_discrete_sequence=[color])
                
                # Extraemos los trazos y los pasamos al subplot
                for trace in temp_fig.data:
                    fig.add_trace(trace, row=1, col=col_idx)
                
                fig.update_yaxes(title_text=f"Número de {y_label}", row=1, col=col_idx)
        
        fig.update_layout(
            title_text=f"{titulo} en r/{sub}",
            bargap=0.05,
            showlegend=False,
            hovermode="x unified"
        )
        fig.show()

def boxplot(lista_dfs: list[pd.DataFrame], metrica: str = 'palabras') -> None:
    """
    Genera un diagrama de caja (boxplot) comparando una métrica 
    específica entre varios subreddits.
    
    Opciones para 'metrica': 'palabras', 'frases', 'palabras_por_frase', 'ttr'
    """
    if not lista_dfs:
        print("⚠️ La lista de DataFrames está vacía.")
        return
        
    # Juntamos todos los DataFrames en uno solo temporalmente para Plotly
    df_combinado = pd.concat(lista_dfs, ignore_index=True)
    
    # Comprobación de seguridad
    if metrica not in df_combinado.columns or 'subreddit' not in df_combinado.columns:
        print(f"⚠️ Error: Faltan columnas necesarias ('{metrica}' o 'subreddit'). Asegúrate de enriquecer e inyectar el subreddit primero.")
        return
        
    # Diccionario de títulos según la métrica elegida
    titulos_metricas = {
        'palabras': 'Longitud del Comentario (Nº de Palabras)',
        'frases': 'Cantidad de Frases por Comentario',
        'palabras_por_frase': 'Complejidad (Palabras por Frase)',
        'ttr': 'Riqueza Léxica (Type-Token Ratio %)'
    }
    
    titulo_eje_y = titulos_metricas.get(metrica, metrica.capitalize())
    
    # Creamos el Boxplot
    fig = px.box(
        df_combinado, 
        x="subreddit", 
        y=metrica, 
        color="subreddit",
        title=f"Distribución de {titulo_eje_y} por Subreddit",
        labels={"subreddit": "Comunidad (r/)", metrica: titulo_eje_y}
    )
    
    fig.update_layout(
        showlegend=False,
        yaxis_type="log"
    )

    fig.show()
    
def plot_top_ngramas(lista_dfs: list[pd.DataFrame], n: int = 2, top_k: int = 15, columna_texto: str = 'body') -> None:
    """
    Extrae los N-gramas más frecuentes de una lista de DataFrames, filtrando stopwords,
    y genera un gráfico de barras horizontales por cada subreddit.
    """
    if not lista_dfs:
        print("⚠️ La lista de DataFrames está vacía.")
        return
        
    # Cargamos la lista de palabras vacías FUERA del bucle para que vaya más rápido
    stop_words = set(stopwords.words('english'))
    # Añadimos "basurilla" típica de Reddit que las stopwords oficiales no pillan
    stop_words.update(['like', 'would', 'could', 'get', 'one', 'people', 'think', 'know', 'really', 'even', 'much','gon', 'na', 'got', 'ta', 'wan'])
    
    # Iteramos por cada DataFrame de la lista
    for df in lista_dfs:
        # Si el DataFrame viene vacío o no tiene texto, saltamos al siguiente
        if df.empty or columna_texto not in df.columns:
            continue
            
        sub_nombre = df.get('subreddit', pd.Series(['Desconocido'])).iloc[0]
        
        todas_palabras = []
        
        # Limpieza y Tokenización fila por fila
        for texto in df[columna_texto].dropna():
            tokens = word_tokenize(str(texto).lower())
            
            # Filtramos palabras alfabéticas que no sean stopwords
            limpios = [word for word in tokens if word.isalpha() and word not in stop_words]
            todas_palabras.extend(limpios)
            
        # Generamos los N-gramas con NLTK
        lista_ngramas = list(ngrams(todas_palabras, n))
        
        if not lista_ngramas:
            print(f"⚠️ No hay suficientes palabras para extraer {n}-gramas en r/{sub_nombre}")
            continue
            
        # Convertimos y contamos
        ngramas_str = [" ".join(gram) for gram in lista_ngramas]
        conteo = Counter(ngramas_str)
        top_ngramas = conteo.most_common(top_k)
        
        # Preparamos los datos para Plotly
        df_plot = pd.DataFrame(top_ngramas, columns=['N-grama', 'Frecuencia'])
        df_plot = df_plot.sort_values(by='Frecuencia', ascending=True) # El más frecuente arriba
        
        tipo_ngram = "Bigramas" if n == 2 else "Trigramas" if n == 3 else f"{n}-gramas"
        
        # Dibujamos el gráfico para este subreddit
        fig = px.bar(
            df_plot, 
            x='Frecuencia', 
            y='N-grama', 
            orientation='h',
            title=f'Top {top_k} {tipo_ngram} en r/{sub_nombre}',
            labels={'Frecuencia': 'Apariciones', 'N-grama': tipo_ngram},
            color='Frecuencia', 
            color_continuous_scale='sunsetdark'
        )
        
        fig.show()
        
def plot_wordclouds(lista_dfs: list[pd.DataFrame], columna_texto: str = 'body') -> None:
    """
    Genera una Nube de Palabras (WordCloud) por cada DataFrame en la lista,
    filtrando las palabras vacías y la 'basurilla' de Reddit.
    """
    if not lista_dfs:
        print("⚠️ La lista de DataFrames está vacía.")
        return

    # Limpiamos de stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update([
        'like', 'would', 'could', 'get', 'one', 'people', 'think', 'know', 'really', 'even', 'much',
        'gon', 'na', 'got', 'ta', 'wan', 'make', 'time', 'good', 'thing', 'see', 'want', 'way', 'say'
    ])

    # Iteramos por cada subreddit
    for df in lista_dfs:
        if df.empty or columna_texto not in df.columns:
            continue
            
        sub_nombre = df.get('subreddit', pd.Series(['Desconocido'])).iloc[0]
        
        # Juntamos TODOS los comentarios de este subreddit en un solo bloque de texto gigante
        texto_completo = " ".join(df[columna_texto].dropna().astype(str).str.lower())
        
        if not texto_completo.strip():
            print(f"⚠️ No hay texto válido en r/{sub_nombre}")
            continue
            
        # Creamos y generamos la Nube de Palabras
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color="white",       
            stopwords=stop_words,
            max_words=100,            # Limitamos a las 100 mejores para no hacer un borrón
            contour_width=3,
            contour_color='steelblue'
        ).generate(texto_completo)
        
        # 5. Dibujamos la imagen generada usando matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f"Nube de Palabras - r/{sub_nombre}", fontsize=18, pad=20)
        plt.axis('off')
        plt.show()

def plot_scatter_longitud_score(lista_dfs: list[pd.DataFrame], columna_longitud: str = 'palabras') -> None:
    """
    Genera un diagrama de dispersión (scatter plot) para visualizar si escribir 
    comentarios más largos garantiza una mayor puntuación (score).
    """
    if not lista_dfs:
        print("⚠️ La lista de DataFrames está vacía.")
        return
        
    # Juntamos todos los DataFrames filtrados en uno solo
    df_combinado = pd.concat(lista_dfs, ignore_index=True)
    
    # Comprobaciones de seguridad
    if columna_longitud not in df_combinado.columns or 'score' not in df_combinado.columns:
        print(f"⚠️ Error: Faltan columnas ('{columna_longitud}' o 'score') en el DataFrame.")
        return
        
    if 'subreddit' not in df_combinado.columns:
        df_combinado['subreddit'] = 'Desconocido'
        
    # Dibujamos el Scatter Plot
    fig = px.scatter(
        df_combinado,
        x=columna_longitud,
        y='score',
        color='subreddit',
        facet_col='subreddit',
        facet_col_wrap=3,
        title='Impacto de la Longitud del Comentario en su Puntuación (Score)',
        labels={
            columna_longitud: 'Longitud del Comentario (Nº de Palabras)',
            'score': 'Puntuación neta (Score)',
            'subreddit': 'Comunidad (r/)'
        },
        opacity=0.6,
        hover_data=['author'],
    )
    
    # Retoques estéticos
    fig.update_layout(
        plot_bgcolor='white',
        showlegend=False,
        legend_title_text='Subreddit',
        hovermode="closest",
        height=600,
        
    )
    
    # Le ponemos una cuadrícula suave para guiar el ojo
    fig.update_xaxes(type='log',showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    fig.show()

def plot_pie(lista_dfs: list[pd.DataFrame],  titulo: str, columna: str = 'language') -> None:
    """
    Genera un gráfico de tarta (pie chart) para visualizar la distribución de una columna categórica.
    """
    if not lista_dfs:
        print("⚠️ La lista de DataFrames está vacía.")
        return

    # Juntamos todos los DataFrames filtrados en uno solo
    df_combinado = pd.concat(lista_dfs, ignore_index=True)

    # Comprobaciones de seguridad
    if columna not in df_combinado.columns:
        print(f"⚠️ Error: La columna '{columna}' no existe en el DataFrame.")
        return

    # Contamos las ocurrencias de cada categoría
    conteo = df_combinado[columna].value_counts()

    # Dibujamos el gráfico de tarta
    plt.figure(figsize=(8, 8))
    fig = px.pie(
        names=conteo.index, 
        values=conteo.values,
        title=titulo)
    
    fig.show()

def barplot(lista_dfs: list[pd.DataFrame], columna: str='author', titulo: str='Top Autores en', top_k: int=15, labels: dict={'x': 'Total', 'y': 'Author'}) -> None:
    
    subs_dict = {}
    for df in lista_dfs:
        sub_name = df['subreddit'].iloc[0]
        if sub_name not in subs_dict:
            subs_dict[sub_name] = {}
        
        tipo = 'subs' if 'title' in df.columns else 'comms'
        subs_dict[sub_name][tipo] = df

    # Creamos la figura doble por cada subreddit
    for sub, dfs in subs_dict.items():
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=(f'Submissions (Hilos)', f'Comentarios'),
            horizontal_spacing=0.15 # Un poco más de espacio para los nombres de usuario
        )

        # Configuración para cada lado (Tipo, Columna, Escala de color)
        config = [
            ('subs', 1, 'Oranges'),
            ('comms', 2, 'Blues')
        ]

        for tipo, col_idx, color_scale in config:
            if tipo in dfs:
                df_actual = dfs[tipo]
                # Calculamos el conteo y ordenamos para que el mayor quede arriba en el gráfico horizontal
                conteo = df_actual[columna].value_counts().head(top_k).sort_values(ascending=True)

                # Creamos el barplot temporal con Plotly Express
                temp_fig = px.bar(
                    conteo,
                    x=conteo.values,
                    y=conteo.index,
                    orientation='h',
                )
                
                # Trasplantamos los trazos
                for trace in temp_fig.data:
                    trace.marker.color = conteo.values
                    trace.marker.colorscale = color_scale
                    trace.marker.showscale = False 
                    fig.add_trace(trace, row=1, col=col_idx)
                
                fig.update_xaxes(title_text="Total publicaciones", row=1, col=col_idx)
                fig.update_layout(title_text=f"{titulo} en r/{sub} (Top {top_k})", showlegend=False)
        
        fig.show()

## -- MÉTRICAS --

def calcular_metricas_lexicas(texto: str) -> dict:
    """
    Calcula métricas de volumen y complejidad de un texto.
    """
    # Si el comentario está vacío o es nulo, devolvemos ceros
    if not isinstance(texto, str) or not texto.strip():
        return {'tokens': 0, 'palabras': 0, 'frases': 0, 'palabras_por_frase': 0.0, 'ttr': 0.0}

    # 1. Tokenización (separa palabras, signos de puntuación, símbolos)
    tokens = word_tokenize(texto.lower())
    
    # 2. Separación de frases (detecta puntos, exclamaciones, etc.)
    frases = sent_tokenize(texto)

    # 3. Extraer solo las palabras (filtramos puntos, comas, emojis...)
    palabras = [t for t in tokens if t.isalnum()]

    num_tokens = len(tokens)
    num_palabras = len(palabras)
    num_frases = len(frases)

    # 4. Palabras por frase (Longitud media de las oraciones)
    palabras_por_frase = num_palabras / num_frases if num_frases > 0 else 0

    # 5. Riqueza Léxica (Type-Token Ratio)
    # Mide cuántas palabras ÚNICAS hay frente al total de palabras.
    tipos_unicos = len(set(palabras))
    ttr = (tipos_unicos / num_palabras * 100) if num_palabras > 0 else 0

    return {
        'tokens': num_tokens,
        'palabras': num_palabras,
        'frases': num_frases,
        'palabras_por_frase': round(palabras_por_frase, 2),
        'ttr': round(ttr, 2)
    }

def metricas_a_df(df_list: list[pd.DataFrame], columna_texto: str = 'body') -> list[pd.DataFrame]:
    """
    Aplica las métricas léxicas a una lista de DataFrames y añade los resultados como nuevas columnas.
    """
    dfs_procesados = [] # Creamos nuestra lista recolectora
    
    for df in df_list:
        # Si el df está vacío o no tiene la columna, lo metemos tal cual y pasamos al siguiente
        if df.empty or columna_texto not in df.columns:
            dfs_procesados.append(df)
            continue
            
        sub_nombre = df['subreddit'].iloc[0] if 'subreddit' in df.columns else 'Desconocido'
        print(f"Calculando métricas para {len(df)} filas de r/{sub_nombre}...")
        
        # Aplicamos la lógica matemática
        metricas_serie = df[columna_texto].apply(calcular_metricas_lexicas)
        df_metricas = pd.DataFrame(metricas_serie.tolist(), index=df.index)
        
        # Juntamos, guardamos en la lista recolectora y el bucle sigue girando
        df_final = pd.concat([df, df_metricas], axis=1)
        dfs_procesados.append(df_final)
        
    # Cuando el bucle termina con la lista de DataFrames, devolvemos la lista completa
    print("✅ ¡Todas las métricas calculadas!")
    return dfs_procesados

import pandas as pd

def calcular_estadisticos_corpus(lista_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Calcula estadísticas generales agregadas por subreddit a partir de 
    una lista de DataFrames que ya han sido enriquecidos con métricas léxicas.
    """
    estadisticos = []
    
    for df in lista_dfs:
        if df.empty:
            continue
            
        # Extraemos el nombre del subreddit
        sub_nombre = df['subreddit'].iloc[0] if 'subreddit' in df.columns else 'Desconocido'
        
        # Comprobación de seguridad: vemos si las columnas existen
        if 'palabras' not in df.columns:
            print(f"⚠️ El DataFrame de r/{sub_nombre} no tiene las métricas calculadas aún.")
            continue
            
        # Totales (Volumen bruto del subreddit)
        total_comentarios = len(df)
        total_palabras = df['palabras'].sum()
        total_tokens = df['tokens'].sum()
        total_frases = df['frases'].sum()
        
        # Medias (Para entender el comportamiento del usuario promedio)
        media_palabras = round(df['palabras'].mean(), 2)
        media_frases = round(df['frases'].mean(), 2)
        media_ttr = round(df['ttr'].mean(), 2) # TTR medio por comentario
        
        # Guardamos la fila de este subreddit
        estadisticos.append({
            'Subreddit': f"r/{sub_nombre}",
            'Comentarios': total_comentarios,
            'Total Palabras': total_palabras,
            'Total Tokens': total_tokens,
            'Media Palabras/Coment.': media_palabras,
            'Media Frases/Coment.': media_frases,
            'Riqueza Léxica Media (TTR %)': media_ttr
        })
        
    # Convertimos la lista de diccionarios en un DataFrame ordenado
    df_resumen = pd.DataFrame(estadisticos)
    
    df_resumen = df_resumen.sort_values(by='Total Palabras', ascending=False).reset_index(drop=True)
    
    return df_resumen