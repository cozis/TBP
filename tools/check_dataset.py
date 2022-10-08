
# Controlla che il dataset sia stato generato senza
# mancare immagini.

import os
from os import path
import json

old_dataset_path = './Dataset'
new_dataset_path = './Faces'

def check_sub_dir(sub_dir_path, images_per_video):

    file_names = [file for file in os.listdir(path.join(new_dataset_path, sub_dir_path))]

    file_indices = []

    for file_name in file_names:

        temp = file_name[6:-4]
        temp = temp.split('_face_')

        temp[0] = int(temp[0])
        temp[1] = int(temp[1])

        file_indices.append((temp[0], temp[1]))

    file_indices = sorted(file_indices)


    missing_frames = []

    expected_index  = 0
    expected_index2 = 0

    last_missing_video = -1

    for index, index2 in file_indices:


        while index != expected_index or index2 != expected_index2:

            missing_frame_name = path.join(sub_dir_path, f'video_{expected_index}_face_{expected_index2}.png')

            print('Manca il frame', missing_frame_name)


            if len(missing_frames) > 0 and expected_index == missing_frames[-1]['index']:

                missing_frames[-1]['count'] += 1

            else:

                missing_frames.append({'index': expected_index, 'count': 1})

                last_missing_video = expected_index


            expected_index2 += 1

            if expected_index2 == images_per_video:
                expected_index += 1
                expected_index2 = 0


        expected_index2 += 1

        if expected_index2 == images_per_video:
            expected_index += 1
            expected_index2 = 0

    video_names = [f for f in os.listdir(path.join(old_dataset_path, sub_dir_path)) if path.isfile(path.join(old_dataset_path, sub_dir_path, f))]
    video_names = sorted(video_names)

    for missing_frame in missing_frames:

        missing_frame['video'] = video_names[missing_frame['index']]

    return missing_frames


fake_missing_list = check_sub_dir('fake', 10)
real_missing_list = check_sub_dir('real', 100)

with open('fake_missing.json', 'w') as outfile:
    json.dump(fake_missing_list, outfile)

with open('real_missing.json', 'w') as outfile:
    json.dump(real_missing_list, outfile)

for item in fake_missing_list:

    print(item)

for item in real_missing_list:

    print(item)

print('I volti mancanti in /fake sono', len(fake_missing_list))
print('I volti mancanti in /real sono', len(real_missing_list))
