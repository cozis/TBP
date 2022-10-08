
import os
from os import path
import json
import shutil

from main_detect_face import main_detect_face
from main_extract_face import main_extract_face

tmp_path = './tmp'
old_dataset_path = './Dataset'
new_dataset_path = './Faces'

def extract_faces_from_video(source_video_path, source_video_index, 
                             dest_faces_path, faces_per_video, 
                             max_frames_per_video):

    # Crea le cartelle in cui
    # buttare i risultati temporanei

    os.system(f"rm -rf {tmp_path}")
    os.mkdir(tmp_path)
    os.mkdir(path.join(tmp_path, 'result'))

    # File risultato dall'operazione di detect (ed input dell'extract)
    detect_output_file = path.join(tmp_path, 'tmp')


    # L'effettivo heavy lifting

    main_detect_face (filevideo = source_video_path, fileboxes = detect_output_file,          num        = max_frames_per_video)
    main_extract_face(filevideo = source_video_path, fileboxes = detect_output_file + '.npz', fileoutput = path.join(tmp_path, 'result'))

    # Ricava la lista dei nomi dei file che contengono i volti
    faces = [f for f in os.listdir(path.join(tmp_path, 'result'))
             if path.isfile(path.join(tmp_path, 'result', f))]

    print(f'Estratti {len(faces)} volti dal video {video}')

    # Se ci sono abbastanza volti allora si calcola il passo
    # secondo cui prelevare il numero di frame richiesto.
    # I volti sono prelevati in modo da garantire la massima
    # distanza tra gli stessi.
    # Se non vi sono abbastanza frame, allora lo si utilizzano
    # tutti i frame del video.
    if len(faces) >= faces_per_video:

      step = len(faces) // int(faces_per_video)

      faces = faces[:faces_per_video*step:step]

    else:

      print(f"WARNING! Il video {source_video_path} ha meno di {faces_per_video} frame (ne ha {len(faces)})")

    # Copia dei volti estratti all'interno della cartella
    # contenente i risultati finali delle elaborazioni.

    index2 = 0

    for face in faces:

        from_file = path.join(tmp_path, "result", face)
        to_file   = path.join(dest_faces_path, f'video_{source_video_index}_face_{index2}.png')

        shutil.copyfile(from_file, to_file)

        index2 += 1

with open('real_missing.json') as file:

    real_missing = json.loads(file.read())
    # [{'index': int, 'count': int, 'video': string}]

for no, item in enumerate(real_missing):

    index = item['index']
    video = item['video']

    source_video_path = path.join(old_dataset_path, 'real', video)

    extract_faces_from_video(source_video_path, index, './single', 100, 150)

    print(no + 1, 'of', len(real_missing), 'done')
