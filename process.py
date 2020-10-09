import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(filepath):
    df = pd.read_csv(filepath)
    print(df.describe())
    print(df.Label.value_counts())

    # scatter plot
    #x = df.Label
    #y = df.Confidence
    #plt.scatter(x, y)
    #plt.xticks(rotation=90)
    #plt.show()

    # person boxplot
    df_person = df[df.Label == 'person']
    print(df_person.describe())
    #df_person.boxplot(column=['Confidence'])
    #plt.show()

    # chair boxplot
    df_chair = df[df.Label == 'chair']
    print(df_chair.describe())
    #df_chair.boxplot(column=['Confidence'])
    #plt.show()

    # backpack boxplot
    df_bp = df[df.Label == 'backpack']
    print(df_bp.describe())
    #df_bp.boxplot(column=['Confidence'])
    #plt.show()

    # sports ball boxplot
    df_sb = df[df.Label == 'sports ball']
    print(df_sb.describe())
    #df_sb.boxplot(column=['Confidence'])
    #plt.show()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--file', nargs='?', type=str, default="",
        help = 'File path of csv'
    )

    FLAGS = parser.parse_args()

    if "file" in FLAGS:
        plot(FLAGS.file)
    else:
        print("Must specify csv file path")