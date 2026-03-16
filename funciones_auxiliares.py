
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
from datetime import datetime, UTC


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


def extract_submissions(filepath, subreddit, n_submissions=3, min_comments=250):
    """
    Extrae las primeras N submissions de un subreddit con al menos min_comments comentarios.
    Captura TODOS los atributos del dump.
    ¡OJO! Para la práctica 2 se deberá realizar un procesamiento de qué subreddits se pretenden seleccionar
    teniendo en cuenta distintas características de los posts como la fecha, información sobre los sumbissions, etc.
    Además, no se deberán almacenar todos los atributos, sino los más relevantes para los usuarios.
    """
    submissions = []
    subreddit_lower = subreddit.lower()

    print(f"🔍 Buscando submissions en r/{subreddit} con ≥{min_comments} comentarios...")

    count = 0
    scanned = 0

    for obj in stream_zst_file(filepath):
        if obj.get('subreddit', '').lower() != subreddit_lower:
            continue

        scanned += 1
        num_comments = obj.get('num_comments', 0) or 0

        if num_comments < min_comments:
            continue

        # Copiar TODOS los atributos originales
        submission = obj.copy()

        # Añadir campos calculados
        submission['name'] = f"t3_{obj.get('id')}"
        submission['created_datetime'] = datetime.fromtimestamp(
            obj.get('created_utc', 0), UTC
        ).isoformat() if obj.get('created_utc') else None
        submission['comments'] = []  # Se llenará después con los comentarios

        submissions.append(submission)
        count += 1
        print(f"  ✓ [{count}/{n_submissions}] ({num_comments} comments) {submission.get('title', '')[:50]}...")

        if count >= n_submissions:
            break

    print(f"\n  Escaneadas: {scanned} del subreddit")
    print(f"  Encontradas: {len(submissions)} con ≥{min_comments} comentarios\n")
    return submissions


def extract_comments_for_submissions(filepath, submissions, num_comments=10):
    """
    Extrae los primeros N comentarios para cada submission.
    Captura TODOS los atributos del dump.
    """
    if num_comments <= 0:
        print("⏭️  num_comments=0, saltando búsqueda de comentarios")
        return

    if not submissions:
        print("⏭️  Sin submissions")
        return

    submission_map = {s['name']: s for s in submissions}
    comment_count = {s['name']: 0 for s in submissions}
    pending = set(submission_map.keys())

    print(f"🔍 Buscando hasta {num_comments} comentarios por submission...")

    total_comments = 0

    for obj in stream_zst_file(filepath):
        link_id = obj.get('link_id')

        if link_id not in pending:
            continue

        # Copiar TODOS los atributos originales
        comment = obj.copy()

        # Añadir campos calculados
        comment['name'] = f"t1_{obj.get('id')}"
        comment['created_datetime'] = datetime.fromtimestamp(
            obj.get('created_utc', 0), UTC
        ).isoformat() if obj.get('created_utc') else None

        submission_map[link_id]['comments'].append(comment)
        comment_count[link_id] += 1
        total_comments += 1

        if comment_count[link_id] >= num_comments:
            pending.discard(link_id)
            if not pending:
                break

    print(f"  Total: {total_comments} comentarios encontrados\n")
    for s in submissions:
        print(f"  📝 \"{s.get('title', '')[:50]}...\" → {len(s['comments'])} comentarios")
        
