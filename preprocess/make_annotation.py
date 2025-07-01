import pandas as pd
import numpy as np
import os, glob
import argparse

def main(args):
    data = {
    0 : 'assault_frame',
    1 : 'burglary_frame',
    2 : 'kidnap_frame',
    3 : 'robbery_frame',
    4 : 'swoon_frame',
    5 : 'normal_frame'
    }

    fpath = args.path


    assault_files = []
    file_path = f'{fpath}/frame_data/assault_frame'
    for fname in glob.glob(os.path.join(file_path, '*')):
        assault_files.append(fname)
    assault_files.sort()


    burglary_files = []
    file_path = f'{fpath}/frame_data/burglary_frame'
    for fname in glob.glob(os.path.join(file_path, '*')):
        burglary_files.append(fname)
    burglary_files.sort()


    kidnap_files = []
    file_path = f'{fpath}/frame_data/kidnap_frame'
    for fname in glob.glob(os.path.join(file_path, '*')):
        kidnap_files.append(fname)
    kidnap_files.sort()


    robbery_files = []
    file_path = f'{fpath}/frame_data/robbery_frame'
    for fname in glob.glob(os.path.join(file_path, '*')):
        robbery_files.append(fname)
    robbery_files.sort()


    swoon_files = []
    file_path = f'{fpath}/frame_data/swoon_frame'
    for fname in glob.glob(os.path.join(file_path, '*')):
        swoon_files.append(fname)
    swoon_files.sort()


    normal_files = []
    file_path = f'{fpath}/frame_data/normal_frame'
    for fname in glob.glob(os.path.join(file_path, '*')):
        normal_files.append(fname)
    normal_files.sort()



    all_file = {
        0:assault_files,
        1:burglary_files,
        2:kidnap_files,
        3:robbery_files,
        4:swoon_files,
        5:normal_files
    }


    catg = list(data.keys())

    df = []


    for c in catg:
        file = all_file[c]
        for i_file in file:
            frames = []
            for frame in glob.glob(os.path.join(i_file, '*.jpg')):
                frames.append(frame)
            frames.sort()
            for f in frames:
                tmp = {
                    'frame_name':f,
                    'label':c,
                    'category':data[c][:-6]
                }
                df.append(tmp)



    df = pd.DataFrame(df)
    df.to_csv('annotation.csv')


    idx = df[df['label']==0]['Unnamed: 0'].values.tolist()
    ast_idx = np.random.permutation(idx)[:8500].tolist()
    idx = df[df['label']==1]['Unnamed: 0'].values.tolist()
    bgr_idx = np.random.permutation(idx)[:8500].tolist()
    idx = df[df['label']==2]['Unnamed: 0'].values.tolist()
    kdn_idx = np.random.permutation(idx)[:8500].tolist()
    idx = df[df['label']==3]['Unnamed: 0'].values.tolist()
    rby_idx = np.random.permutation(idx)[:8500].tolist()
    idx = df[df['label']==4]['Unnamed: 0'].values.tolist()
    swn_idx = np.random.permutation(idx)[:8500].tolist()
    idx = df[df['label']==5]['Unnamed: 0'].values.tolist()
    nml_idx = np.random.permutation(idx)[:8500].tolist()


    final = ast_idx+bgr_idx+kdn_idx+rby_idx+swn_idx+nml_idx

    new_ = df.iloc[final]

    new_.to_csv('./train_annotation.csv')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')

    args = parser.parse_args()
    main(args)
