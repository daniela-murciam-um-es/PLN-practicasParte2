import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Intentamos descargar silenciosamente. Si ya están, no hace nada.
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"⚠️ Advertencia: No se pudieron verificar los paquetes NLTK ({e}). Si la tokenización falla, ejecuta nltk.download('punkt') manualmente.")

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

def histograma(datos: list[pd.DataFrame], atributo: str = 'created_datetime', ) -> None:
    '''
    Genera histogramas de los dataframes pasados en una lista, sobre el atributo (columna del dataframe) 
    especificado. Distingue si es submission o comment para que quede más claro y se diferencie mejor.
    '''
    for dataframe in datos:
        if 'title' in dataframe: # Es submission
            tipo_dato = 'Submissions (Hilos)'
            y_label = 'Número de Hilos'
            sub = dataframe['subreddit'].iloc[0]
            color = '#FF5700'
        else:
            tipo_dato = 'Comentarios'
            y_label = 'Número de Comentarios'
            sub = dataframe['subreddit'].iloc[0]
            color = '#336699'

        # Crea el histograma del tipo que sea en ese momento 
        fig = px.histogram(dataframe,
               x=atributo,
               title=f'Distribucion temporal de {tipo_dato} en el subreddit: {sub}',
               labels={'created_datetime': 'Fecha y Hora de publicación'},
               color_discrete_sequence=[color])
        
        fig.update_layout(
            yaxis_title=y_label,
            bargap=0.05,
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