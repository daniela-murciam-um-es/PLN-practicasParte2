
"""
=====================
Este ejemplo procesa archivos de un volcado de Reddit en formato .zst para extraer
submissions y sus comentarios de un subreddit específico.
"""

import zstandard as zstd
import json
import argparse
import io
import sys
from pathlib import Path
from datetime import datetime, timedelta, UTC

import zstandard as zstd
import json
import io
from datetime import datetime, timedelta, UTC

# Atributos relevantes para el análisis de texto y relevancia social
ATTRIBUTOS_SUB = ['id', 'title', 'selftext', 'score', 'num_comments', 'created_utc', 'subreddit', 'name']
ATTRIBUTOS_COMM = ['id', 'body', 'score', 'parent_id', 'link_id', 'created_utc', 'controversiality', 'author']

def filter_submission(obj):
    return {clave: valor for clave, valor in obj.items() if clave in ATTRIBUTOS_SUB}

def filter_comment(obj):
    return {clave: valor for clave, valor in obj.items() if clave in ATTRIBUTOS_COMM}


def stream_zst_file(filepath):
	"""
	Genera objetos JSON línea por línea desde un archivo .zst
	"""
	dctx = zstd.ZstdDecompressor(max_window_size=2**31)

	with open(filepath, 'rb') as fh:
		with dctx.stream_reader(fh) as reader:
			text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')

			for line in text_stream:
				line = line.strip()
				if not line:
					continue
				try:
					yield json.loads(line)
				except json.JSONDecodeError:
					continue


def extract_submissions(filepath, subreddits_list, n_submissions=40, min_comments=30):
    subs_buscados = {sub.lower(): [] for sub in subreddits_list}
    # Guardamos el último 'created_utc' guardado para cada subreddit
    last_time_saved = {sub.lower(): 0 for sub in subreddits_list}
    
    # HE CONFIGURADO UN SALTO ALEATORIO, EN ESTE CASO ES UN DIA, PARA QUE NO HAYA VARIOS DEL MISMO DIA
    SALTO = 86400 

    for obj in stream_zst_file(filepath):
        sub = obj.get('subreddit', '').lower()
        
        if sub in subs_buscados and len(subs_buscados[sub]) < n_submissions:
            # Comprobamos si tiene comentarios suficientes
            if obj.get('num_comments', 0) >= min_comments:
                
                # COMPROBACIÓN DE SALTO TEMPORAL
                actual_time = obj.get('created_utc', 0)
                if abs(actual_time - last_time_saved[sub]) > SALTO:
                    
                    # ATRIBUTOS FILTRADOS
                    post_limpio = {
                        'id': obj.get('id'),
                        'name': obj.get('name'), # t3_... (necesario para link_id)
                        'title': obj.get('title'),
                        'selftext': obj.get('selftext'),
                        'score': obj.get('score'),
                        'created_utc': actual_time,
                        'subreddit': sub,
                        'comments': [] # Aquí meteremos los comentarios luego
                    }
                    
                    subs_buscados[sub].append(post_limpio)
                    last_time_saved[sub] = actual_time
                    print(f"Post guardado en r/{sub} ({len(subs_buscados[sub])}/{n_submissions})")

    # Convertimos el diccionario en una lista plana para devolverla
    resultado = []
    for lista in subs_buscados.values():
        resultado.extend(lista)
    return resultado

def extract_comments_for_submissions(filepath, submissions, num_comments=35):
    """
    Busca comentarios para las submissions seleccionadas.
    Pedimos 35 por hilo para tener margen de sobra si hay bots o spam.
    """
    submission_map = {s['name']: s for s in submissions}
    comment_count = {s['name']: 0 for s in submissions}
    pending = set(submission_map.keys())
    
    print(f"🔍 Buscando ~{num_comments} comentarios por hilo...")

    for obj in stream_zst_file(filepath):
        link_id = obj.get('link_id')

        if link_id in pending:
            # Filtrado de atributos
            comment = filter_comment(obj)
            submission_map[link_id]['comments'].append(comment)
            comment_count[link_id] += 1

            if comment_count[link_id] >= num_comments:
                pending.remove(link_id)

        if not pending:
            break
    print("✨ Extracción de comentarios finalizada.")
    