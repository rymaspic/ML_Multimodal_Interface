import sys
from sklearn.model_selection import train_test_split

from Gesture import GestureSet, Sequence, Frame
from classify_nn import classify_nn
from normalize_frames import normalize_frames
from load_gestures import load_gestures

def test_classify_nn(num_frames, ratio):
    """
    Tests classify_nn function. 
    Splits gesture data into training and testing sets and computes the accuracy of classify_nn()
    :param num_frames: the number of frames to normalize to
    :param ratio: percentage to be used for training
    :return: the accuracy of classify_nn()
    """

    gesture_sets = load_gestures()
    norm_gesture_sets = normalize_frames(gesture_sets, num_frames)
    num = len(norm_gesture_sets)
    train_set = []
    test_set = []
    for i in range(int(num * ratio)):
        train_set.append(norm_gesture_sets[i])

    test_set = list(set(gesture_sets) - set(train_set))

    total_num = 0
    correct_num = 0
    for gesture in test_set:
        sequence_num = len(gesture.sequences)
        total_num = total_num + sequence_num
        correct_label = gesture.label
        for sequence in gesture.sequences:
            label = classify_nn(sequence,train_set)
            if label is correct_label:
                correct_num = correct_num + 1

    return correct_num / total_num


if len(sys.argv) != 3:
    raise ValueError('Error! Give normalized frame number and test/training ratio after filename in command. \n'
                     'e.g. python test_nn.py 20 0.4')

num_frames = int(sys.argv[1])
ratio = float(sys.argv[2])

accuracy = test_classify_nn(num_frames, ratio)
print("Accuracy: ", accuracy)