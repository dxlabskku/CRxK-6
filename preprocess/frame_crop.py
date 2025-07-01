import imageio
import glob, os
import argparse

def main(args):
    PATH = args.path
    catg = args.category
    file_path = '{}/{}/{}_crop'
    video = []

    for fname in glob.glob(os.path.join(file_path.format(PATH,catg, catg), '*.mp4')):
        video.append(fname)

    mk_path = './{}'

    video.sort()


    for idx, v in enumerate(video):
        idx_str = str(idx).zfill(3)
        os.mkdir(mk_path.format(idx_str))
        try:
            reader = imageio.get_reader(v)
            cnt = 0
            for frame_number, im in enumerate(reader):
                if frame_number % 3 == 0:
                    str_cnt = str(cnt)
                    imageio.imwrite(f'./{idx_str}/frame_{str_cnt.zfill(4)}.jpg', im)
                    cnt+=1
            print(f'Done with {idx}th video.')
        except:
            print(f'{idx}th video has problem.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--category', type=str, default='kidnap')

    args = parser.parse_args()
    main(args)
