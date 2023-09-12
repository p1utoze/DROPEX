"""
THIS FILE IS TO CONVERT THE ANNOTATIONS FROM JSON TO TXT
THE TXT FILE WILL BE USED TO TRAIN THE YOLOv4 MODEL
THERE ARE 3 DATASET DIRECTORIES: normal, rotated and output
EACH FOLDER CONTAINS 2 FOLDERS: images and annotations
THE IMAGES FOLDER CONTAINS THE IMAGES
THE ANNOTATIONS FOLDER CONTAINS THE JSON FILES and NEW TXT FILES
EACH TEXT FILE CONTAINS THE ANNOTATIONS FOR EACH IMAGE IN THIS FORMAT:
-> class x_center y_center width height
"""
import os
from pathlib import Path
import pandas as pd


def json_to_txt(json_file: Path | str, data_path: Path):
    """
    THIS FUNCTION CONVERTS THE JSON FILE TO TXT FILE
    :param data_path:
    :param json_file: THE JSON FILE | str (Absolute Path)
    :return: NONE
    """
    # with open(json_file) as f:
    #     data = json.load(f)
    data = pd.read_json(json_file, lines=True)
    print(data.columns)

    annot = pd.DataFrame(data['annotation'][0])
    img = pd.DataFrame(data['images'][0])
    annot[['x_min', 'x_max', 'y_min', 'y_max']] = pd.DataFrame(annot['bbox'].tolist(), index=annot.index)
    s = annot.merge(img.drop(['height', 'width'], axis=1), left_on='image_id', right_on='id')

    c = 0
    for img_id in img['id']:
        subset = s.loc[
            s['image_id'] == img_id, ['category_id', 'x_min', 'x_max', 'y_min', 'y_max', 'filename']].reset_index(
            drop=True)
        try:
            filename = data_path / subset.at[0, 'filename']
            filename = filename.with_suffix('.txt')
            subset.drop('filename', axis=1).to_csv(filename, header=None, index=False)
        except KeyError:
            print("Image ID not found in annotations: ", img_id)
            c += 1

    print("Total images not found: ", c)


def main(work_dir: Path = Path.cwd()):
    """
    CREATE THE TXT FILES FOR THE ROTATE_JSON
    ( USE THE PRE_PROCESSED JSON FILES in final/annotations/ if you don't need to slant angle)
    :param work_dir: THE WORKING DIRECTORY | Path
    :return: NONE
    """
    # SET THE WORKING DIRECTORY
    os.chdir(work_dir)

    data_split_dir = os.listdir(os.getcwd())
    print(data_split_dir)
    for folder_index, json_file in enumerate(Path(data_split_dir[0]).glob('*.json'), 1):
        print(data_split_dir[folder_index], json_file)
        # create new dir for txt_annotations
        annotation_dir = Path(data_split_dir[folder_index]) / 'txt_annotations'
        if not annotation_dir.exists():
            annotation_dir.mkdir(parents=True, exist_ok=True)
        # convert json to txt using pandas and save in new annotation_dir
        json_to_txt(json_file, annotation_dir)


if __name__ == '__main__':
    main()
