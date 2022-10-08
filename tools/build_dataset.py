
import os
from os import path
import shutil

from main_detect_face import main_detect_face
from main_extract_face import main_extract_face

#???
state_file  = './state.txt'

# Cartella temporanea dove buttare tutti i
# file temporanei
tmp_path = './tmp'

# Percorso del dataset di video da elaborare
old_dataset_path = './Dataset'

# Percorso del dataset di facce da produrre
new_dataset_path = './Faces'

last_processed_sub_folder = 'fake'
last_processed_video_index = 0

if path.isdir(new_dataset_path):

  if path.isfile(state_file):

    # Il dataset è già stato generato (completamente o parzialmente) ed
    # è stato trovato il file di stato. Possiamo continuare da dove eravamo
    # rimasti.

    with open(state_file) as file:

      data = file.read()

      last_processed_sub_folder  = data.split(' ')[0]
      last_processed_video_index = int(data.split(' ')[1]) + 1

    print(f'LOG: Il dataset è già stato generato in {new_dataset_path} ed è stato trovato il file di stato {state_file}.\nL\'estrazione partirà da dove era stata sospesa')

  else:

    # Il dataset è già stato generato ma non è stato trovato alcun file
    # di stato. Il dataset verrà cancellato e rigenerato.

    print(f'LOG: Il dataset è già stato generato precedentemente in {new_dataset_path} ma non è stato trovato il file di stato {state_file}. Il dataset sarà cancellato e rigenerato')

    # Genera la struttura di cartelle del nuovo dataset.
    os.system(f"rm -rf {new_dataset_path}")
    os.mkdir(new_dataset_path)

else:

  # La cartella del dataset non è stata già generata in esecuzioni
  # precedenti dello script.

  print(f'LOG: Creando il dataset in {new_dataset_path}')

  # Genera la struttura di cartelle del
  # nuovo dataset.
  os.mkdir(new_dataset_path)

def extract_faces_from_dataset_subfolder(old_dataset_sub_dir, faces_per_video, max_frames_per_video,
                                         video_to_process_count = -1, first_video_to_process = 0):

  # ==================================================================== #
  # Itera sui file video in [old_dataset_sub_dir] del vecchio dataset    #
  # ed estrai [faces_per_video] volti (o meno, se non sono abbastanza).  #
  # ==================================================================== #

  # Crea nel nuovo dataset la sottocartella di destinazione
  # delle facce estratte.

  subdir_path = path.join(new_dataset_path, old_dataset_sub_dir)

  if not path.isdir(subdir_path):
    os.mkdir(subdir_path)


  # Elenco di video dai quali estrarre i volti
  video_names = [f for f in os.listdir(path.join(old_dataset_path, old_dataset_sub_dir)) if path.isfile(path.join(old_dataset_path, old_dataset_sub_dir, f))]
  video_names = sorted(video_names)

  # A noi però interessa solo il sottoinsieme di video che parte da
  # quello con indice [first_video_to_process] ed i [video_to_process_count]-1
  # successivi.

  if video_to_process_count < 0:
    video_to_process_count = len(video_names)

  video_names = video_names[first_video_to_process:first_video_to_process + video_to_process_count]


  # Comincia l'elaborazione

  index = first_video_to_process

  print(f'Comincio l\'estrazione dalla sottocartella {old_dataset_sub_dir} a partire dal video con indice {index}\n')

  for video in video_names:

    # Crea le cartelle in cui
    # buttare i risultati temporanei

    os.system(f"rm -rf {tmp_path}")
    os.mkdir(tmp_path)
    os.mkdir(path.join(tmp_path, 'result'))


    # Percorso assoluto di questo video
    video_path = path.join(old_dataset_path, old_dataset_sub_dir, video)

    # File risultato dall'operazione di detect (ed input dell'extract)
    detect_output_file = path.join(tmp_path, 'tmp')


    # L'effettivo heavy lifting

    main_detect_face (filevideo = video_path, fileboxes = detect_output_file,          num        = max_frames_per_video)
    main_extract_face(filevideo = video_path, fileboxes = detect_output_file + '.npz', fileoutput = path.join(tmp_path, 'result'))

    #os.system(f"python {detect_script_path}  --input {video_path} --boxes {detect_output_file} --num {max_frames_per_video}")
    #os.system(f"python {extract_script_path} --input {video_path} --boxes {detect_output_file}.npz --output {path.join(tmp_path, 'result')}")


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

      print(f"WARNING! Il video {video_path} ha meno di {faces_per_video} frame (ne ha {len(faces)})")

    # Copia dei volti estratti all'interno della cartella
    # contenente i risultati finali delle elaborazioni.

    index2 = 0

    for face in faces:

        from_file = path.join(tmp_path, "result", face)
        to_file   = path.join(new_dataset_path, old_dataset_sub_dir, f'video_{index}_face_{index2}.png')

        shutil.copyfile(from_file, to_file)

        index2 += 1

    with open(state_file, 'w') as file:
        file.write(f'{old_dataset_sub_dir} {index}')

    index += 1

# Partenza delle elaborazioni sui video Fake e Real

if last_processed_sub_folder == 'fake':
  extract_faces_from_dataset_subfolder('fake', 10,  15,  -1, last_processed_video_index)
  extract_faces_from_dataset_subfolder('real', 100, 150, -1, 0)
else:
  extract_faces_from_dataset_subfolder('real', 100, 150, -1, last_processed_video_index)
