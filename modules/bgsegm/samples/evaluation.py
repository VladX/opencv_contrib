import argparse
import cv2
import glob
import numpy as np
import os
import time


ALGORITHMS_TO_EVALUATE = [(cv2.bgsegm.createBackgroundSubtractorMOG, 'MOG'), (cv2.bgsegm.createBackgroundSubtractorGMG, 'GMG'), (cv2.bgsegm.createBackgroundSubtractorCNT, 'CNT'), (cv2.bgsegm.createBackgroundSubtractorLSBP, 'LSBP')]


def contains_relevant_files(root):
    return os.path.isdir(os.path.join(root, 'groundtruth')) and os.path.isdir(os.path.join(root, 'input'))


def find_relevant_dirs(root):
    relevant_dirs = []
    for d in sorted(os.listdir(root)):
        d = os.path.join(root, d)
        if os.path.isdir(d):
            if contains_relevant_files(d):
                relevant_dirs += [d]
            else:
                relevant_dirs += find_relevant_dirs(d)
    return relevant_dirs


def load_sequence(root):
    gt_dir, frames_dir = os.path.join(root, 'groundtruth'), os.path.join(root, 'input')
    gt = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    f = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
    gt = np.uint8(map(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), gt))
    f = np.uint8(map(lambda x: cv2.imread(x, cv2.IMREAD_COLOR), f))
    assert(gt.shape[0] == f.shape[0])
    return gt, f


def evaluate_algorithm(gt, frames, algo):
    bgs = algo()
    mask = []
    t_start = time.time()

    for i in range(gt.shape[0]):
        mask.append(bgs.apply(frames[i]))

    average_duration = (time.time() - t_start) / gt.shape[0]
    average_precision, average_recall, average_f1, average_accuracy = [], [], [], []

    for i in range(gt.shape[0]):
        roi = ((gt[i] == 255) | (gt[i] == 0))
        if roi.sum() > 0:
            gt_answer, answer = gt[i][roi], mask[i][roi]

            tp = ((answer == 255) & (gt_answer == 255)).sum()
            tn = ((answer == 0) & (gt_answer == 0)).sum()
            fp = ((answer == 255) & (gt_answer == 0)).sum()
            fn = ((answer == 0) & (gt_answer == 255)).sum()

            precision = 0.0 if tp + fp == 0 else float(tp) / (tp + fp)
            recall = 0.0 if tp + fn == 0 else float(tp) / (tp + fn)
            f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
            accuracy = float(tp + tn) / (tp + tn + fp + fn)

            average_precision.append(precision)
            average_recall.append(recall)
            average_f1.append(f1)
            average_accuracy.append(accuracy)

    return average_duration, np.mean(average_precision), np.mean(average_recall), np.mean(average_f1), np.mean(average_accuracy)


def evaluate_on_sequence(seq):
    gt, frames = load_sequence(seq)
    print('=== %s:%s ===' % (os.path.basename(os.path.dirname(seq)), os.path.basename(seq)))

    for algo, algo_name in ALGORITHMS_TO_EVALUATE:
        print('Algorithm name: %s' % algo_name)
        sec_per_step, precision, recall, f1, accuracy = evaluate_algorithm(gt, frames, algo)
        print('Average accuracy: %.3f' % accuracy)
        print('Average precision: %.3f' % precision)
        print('Average recall: %.3f' % recall)
        print('Average F1: %.3f' % f1)
        print('Average sec. per step: %.4f' % sec_per_step)
        print('')


def main():
    parser = argparse.ArgumentParser(description='Evaluate all background subtractors using Change Detection 2014 dataset')
    parser.add_argument('--dataset_path', help='Path to the directory with dataset. It may contain multiple inner directories. It will be scanned recursively.', required=True)

    args = parser.parse_args()
    dataset_dirs = find_relevant_dirs(args.dataset_path)
    assert len(dataset_dirs) > 0, ("Passed directory must contain at least one sequence from the Change Detection dataset. There is no relevant directories in %s. Check that this directory is correct." % (args.dataset_path))

    for seq in dataset_dirs:
        evaluate_on_sequence(seq)


if __name__ == '__main__':
    main()
