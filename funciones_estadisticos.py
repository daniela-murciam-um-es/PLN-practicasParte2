import pandas as pd
import json
from datetime import datetime
import plotly.express as px

def separar_dataframes(archivo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
	'''
	Esta función separa el archivo json del subreddit que se pase como atributo en dos dataframes de pandas
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

def histograma(datos: list[pd.DataFrame], atributo: str, ):
	'''
	Esta función genera histogramas de los dataframes pasados en una lista, sobre el atributo (columna del dataframe) 
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