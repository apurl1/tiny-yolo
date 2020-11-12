from os import path, makedirs
import pandas as pd
import numpy as np
import re
import os
from PIL import Image
import xml.etree.ElementTree as ET
from Get_File_Paths import GetFileList, ChangeToOtherMachine

def convert_vott_csv_to_yolo(vott_df, labeldict, path="", target_name="data_train.txt", abs_path=True):
    # Encode labels according to labeldict if code's don't exist
    if not "code" in vott_df.columns:
        vott_df["code"] = vott_df["label"].apply(lambda x: labeldict[x])
    # Round float to ints
    for col in vott_df[["xmin", "ymin", "xmax", "ymax"]]:
        vott_df[col] = (vott_df[col]).apply(lambda x: round(x))

    # Create Yolo Text file
    last_image = ""
    txt_file = ""

    for index, row in vott_df.iterrows():
        if not last_image == row["image"]:
            if abs_path:
                txt_file += "\n" + row["image_path"] + " "
            else:
                txt_file += "\n" + os.path.join(path, row["image"]) + " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        else:
            txt_file += " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        last_image = row["image"]
    file = open(target_name, "w")
    file.write(txt_file[1:])
    file.close()
    return True

def csv_from_xml(directory, path_name=""):
    # First get all images and xml files from path and its subfolders
    image_paths = GetFileList(directory, ".jpg")
    xml_paths = GetFileList(directory, ".xml")
    result_df = pd.DataFrame()
    data = {'image': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'label': [], 'code': [], 'image_path': []}
    for image in image_paths:
        target_filename = os.path.join(path_name, image) if path_name else image
        source_filename = os.path.join(directory, image)
        y_size, x_size, _ = np.array(Image.open(source_filename)).shape
        source_xml = image.replace(".jpg", ".xml")
        #print(source_xml)
        tree = ET.parse(source_xml)
        root = tree.getroot()
        for item in root.iter('bndbox'):
            if item.find('xmin') is None:
                break
            #print("found")
            xmin = item.find('xmin').text
            ymin = item.find('ymin').text
            xmax = item.find('xmax').text
            ymax = item.find('ymax').text
            data["xmin"].append(float(xmin))
            data["ymin"].append(float(ymin))
            data["xmax"].append(float(xmax))
            data["ymax"].append(float(ymax))
            data["label"].append("chair")
            data["code"].append(0)
            data["image_path"].append(target_filename)
            data["image"].append(os.path.basename(target_filename))
    result_df = pd.DataFrame(data)
    return result_df

if __name__ == "__main__":
    directory = "/Users/rupaln/Documents/rupaln/uiuc/research/tiny-yolo/training/suitcase-voc"
    
    label_names = ["suitcase",]
    labeldict = dict(zip(label_names, list(range(1))))
    convert_vott_csv_to_yolo(
        csv_from_xml(directory, "/Users/rupaln/Documents/rupaln/uiuc/research/tiny-yolo/training/suitcase-converted"), labeldict
    )