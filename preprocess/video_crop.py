"""
pip install --trusted-host pypi.python.org moviepy
pip install imageio-ffmpeg
pip install beautifulsoup4
pip install lxml
"""

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from bs4 import BeautifulSoup
import glob, os
import argparse

def to_sec(time):
    hms = time.split(':')
    if len(hms) == 3:
        min_ = int(hms[1])*60
        sec = min_ + float(hms[2])
    elif len(hms) == 2:
        min_ = int(hms[0])*60
        sec = min_ + float(hms[1])
    else:
        sec = int(hms[0])
    return sec

def main(args):
    PATH = args.path
    catg = args.category
    std = args.stride
    cnt = args.index

    annotation = []
    video = []


    file_path = '{}/{}/{}_video'
    for fname in glob.glob(os.path.join(file_path.format(PATH,catg, catg), '*.xml')):
        annotation.append(fname)

    for fname in glob.glob(os.path.join(file_path.format(PATH,catg, catg), '*.mp4')):
        video.append(fname)

    annotation.sort()
    video.sort()

    for idx in range(0, len(annotation), std):
        with open(annotation[idx], 'r') as f:
            data = f.read()

        soup = BeautifulSoup(data, "xml")
        e_time = soup.find('starttime').text
        duration = 10

        e_sec = to_sec(e_time) -5
        s_sec = e_sec - duration
        s_sec = round(s_sec, 1)
        print(f's_time: {s_sec} \nduration: {duration} \ne_time: {e_sec}')
        num_str = str(cnt)
        ffmpeg_extract_subclip(video[idx], s_sec, e_sec, targetname=f"normal{num_str.zfill(3)}.mp4")
        cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--category', type=str, default='kidnap')
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--index', type=int, default=71)
    
    args = parser.parse_args()
    main(args)
