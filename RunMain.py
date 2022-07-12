import argparse
import numpy as np
import tracker
from AIDetector_pytorch import Detector
import imutils
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(args):
    func_status = {}
    func_status['headpose'] = None
    name = 'RunMain'
    detector = Detector()
    cap = cv2.VideoCapture(args.video_path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000 / fps)
    size = None
    videoWriter = None

    while True:
        # try:
        _, im = cap.read()
        if im is None:
            break

        ###此方法不显示对象label
        result = detector.feedCap(im, func_status)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
        videoWriter.write(result)
        cv2.imshow(name, result)

        cv2.waitKey(t)
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='video/113355.mp4', help='视频默认地址')
    args = parser.parse_args()
    main(args)