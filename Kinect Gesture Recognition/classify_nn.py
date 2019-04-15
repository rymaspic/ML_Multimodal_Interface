import operator
import math
from normalize_frames import normalize_frames
from scipy.spatial import distance


def classify_nn(test_sequence, training_gesture_sets):
    """
    Classify test_sequence using nearest neighbors
    :param test_gesture: Sequence to classify
    :param training_gesture_sets: training set of labeled gestures
    :return: a classification label (an integer between 0 and 8)
    """
    num_frames = len(test_sequence.frames)
    training_gesture_sets = normalize_frames(training_gesture_sets,num_frames)

    distances = []
    for gesture in training_gesture_sets:
        sum_distance = 0
        for sequence in gesture.sequences:
            for i in range(num_frames):
                train_Frame = sequence.frames[i]
                test_Frame = test_sequence.frames[i]

                dist = distance.euclidean(train_Frame.frame,test_Frame.frame)
                # distance = math.sqrt(sum([(a - b) ** 2 for a, b in
                #                           zip(sequence.frames[i], test_sequence.frames[i])]))
                sum_distance = sum_distance + dist
        sum_distance = sum_distance / len(gesture.sequences) # average l2 distance
        distances.append(sum_distance)

    label = distances.index(min(distances))

    return label
