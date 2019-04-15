from Gesture import GestureSet, Sequence
from scipy import signal

def normalize_frames(gesture_sets, num_frames):
    """
    Normalizes the number of Frames in each Sequence in each GestureSet
    :param gesture_sets: the list of GesturesSets
    :param num_frames: the number of frames to normalize to
    :return: a list of GestureSets where all Sequences have the same number of Frames
    """
    #print(gesture_sets[0].label)
    # print("original")
    # print(len(gesture_sets))
    # https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
    normalized_sets = []
    for gesture in gesture_sets:
        new_sequences = []
        for sequence in gesture.sequences:
            new_frames = []
            current_num_frame = len(sequence.frames)
            f = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
            indices_list = f(num_frames,current_num_frame)
            #print(list(current_num_frame))
            #print(indices_list)
            for i in indices_list:
                new_frames.append(sequence.frames[i])
            new_sequence = Sequence(new_frames,sequence.label)
            new_sequences.append(new_sequence)
        normalized_sets.append(GestureSet(new_sequences,gesture.label))

    #print("new" + str(len(normalized_sets)))

    return normalized_sets


    #raise NotImplementedError("Your Code Here")
