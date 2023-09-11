"""
THIS FILE IS TO CONVERT THE ANNOTATIONS FROM JSON TO TXT
THE TXT FILE WILL BE USED TO TRAIN THE YOLOv4 MODEL
THERE ARE 3 DATASET FOLDERS: normal, rotated and output
EACH FOLDER CONTAINS 2 SUBFOLDERS: images and annotations
THE IMAGES FOLDER CONTAINS THE IMAGES
THE ANNOTATIONS FOLDER CONTAINS THE JSON FILES and NEW TXT FILES
EACH TEXT FILE CONTAINS THE ANNOTATIONS FOR EACH IMAGE IN THIS FORMAT:
>>> class x_center y_center width height
"""
import json
import os
from pathlib import Path


def json_to_txt(json_file: Path, data_path: str):
    """
    THIS FUNCTION CONVERTS THE JSON FILE TO TXT FILE
    :param json_file: THE JSON FILE
    :param txt_file: THE TXT FILE
    :return: NONE
    """
    with open(json_file) as f:
        data = json.load(f)
    print(data.keys())

    for file in Path(data_path).glob('*.jpg'):
        # print(file)
        txt_file = file.with_suffix('.txt')

        with open(txt_file, 'w') as f:
            class_id = data['annotation'][0]['category_id']
            bbox = data['annotation'][0]['bbox']
        #         x_center = obj['relative_coordinates']['center_x']
        #         y_center = obj['relative_coordinates']['center_y']
        #         width = obj['relative_coordinates']['width']
        #         height = obj['relative_coordinates']['height']
        #         class_id = obj['class_id']
        #         f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

def main():
    """
    CREATE THE TXT FILES FOR THE NORMAL DATASET
    :return: NONE
    """
    # print(os.getcwd())
    os.chdir('datasets/output/')

    # print(os.getcwd())
    data_split_dir = os.listdir(os.getcwd())
    print(data_split_dir)
    for folder_index, json_file in enumerate(Path(data_split_dir[0]).glob('*.json'), 1):
        print(data_split_dir[folder_index], json_file)
        # create new dir for txt_annotations
        annotation_dir = Path(data_split_dir[folder_index]) / 'txt_annotations'
        if not annotation_dir.exists():
            annotation_dir.mkdir(parents=True, exist_ok=True)

        json_to_txt(json_file, data_split_dir[folder_index])
    #

    #     txt_file = json_file.with_suffix('.txt')
    #     json_to_txt(json_file, txt_file)
    #     print(f'Created {txt_file}')


if __name__ == '__main__':
    main()

