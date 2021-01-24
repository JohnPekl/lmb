"""Create crosstrack.png plot."""

"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import time
import lmb
import cv2
from os import path
import collections

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_mot(relpath='../MOT20-04/'):
    names = collections.defaultdict(list)
    detections = collections.defaultdict(list)
    # Store the image names.
    for file in os.listdir(path.join(relpath, 'img1')):
        if file.endswith('.jpg'):
            name, extension = file.split('.')
            names[int(name)] = file
    # Load the detections.
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>
    with open(path.join(relpath, 'det/det.txt'), mode='r') as file:
        for line in file:
            frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y = map(int, line.split(',')[:-1])
            # frame,col,row,width,height=map(int,line.split(',')[:-1])
            detections[frame].append(np.array([bb_left, bb_top, bb_width, bb_height]))  # ((0,2) (1,3)

    return names, detections


def draw():
    """Create plot."""
    params = lmb.Parameters()
    params.N_max = 50000
    params.kappa = lmb.models.UniformClutter(0.0001)
    params.init_target = lmb.DefaultTargetInit(0.1, 1, 1)
    params.r_lim = 0.00104  # 0.0010416666666666667
    params.nstd = 10
    tracker = lmb.LMB(params)
    sensor = lmb.sensors.EyeOfMordor()
    sensor.lambdaB = 0.1

    names, detections = read_mot()

    for frame in range(min(names.keys()), max(names.keys())):
        start = time.time()
        if frame > 1:
            tracker.predict(1)
        reports = {lmb.GaussianReport(
            # np.random.multivariate_normal(t[0:2], np.diag([0.01] * 2)),  # noqa
            (obs[:2] + obs[2:] / 2.0),
            np.eye(2) * 5,
            lmb.models.position_measurement,
            i)
            for i, obs in enumerate(detections[frame])}

        this_scan = lmb.Scan(sensor, reports)
        tracker.register_scan(this_scan)
        fps = time.time() - start
        print('frame:', frame, datetime.now().strftime("%H:%M:%S"))

        img = cv2.imread(path.join('../MOT20-04/img1', names[frame]))
        targets = tracker.query_targets()
        print('enof_targets %s, nof_targets %s, detection(len) %s' % (tracker.enof_targets(),
                                                                      tracker.nof_targets(), len(detections[frame])))
        img = cv2.putText(img, 'Frame {}'.format(frame) + ', FPS:{}'.format(round(1 / fps, 2)),
                            org=(1145, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2)
        for t in targets:
            ty, r, x, P = t.history[-1]
            img = cv2.circle(img, (int(x[0]), int(x[1])), radius=10, color=(255, 255, 255), thickness=-1)
            img = cv2.putText(img, str(t.id),
                              org=(int(x[0]), int(x[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65,
                              color=(0, 255, 255), thickness=1)
        for t in detections[frame]:
            bbox = t.astype(int)
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(0, 0, 0),
                                thickness=2)
        cv2.imshow('Image', img)
        cv2.imwrite('../MOT20-04/output/' + str(frame) + '.jpg', img)
        cv2.waitKey(1)


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    draw()


if __name__ == '__main__':
    main(*sys.argv[1:])
