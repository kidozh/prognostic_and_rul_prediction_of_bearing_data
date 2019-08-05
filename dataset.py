import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
# from keras.utils import to_categorical

XJTU_DATA_PATH = r"E:\IDM_download\XJTU-SY_Bearing_Datasets"

CONDITION_DATA_FOLDER_LIST = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]

CONDITION_LABEL = [[35, 12], [37.5, 11], [40, 10]]

CONDITION_MINUTES_NUMBER = [[123, 161, 158, 122, 52], [491, 161, 533, 42, 339], [2538, 2496, 371, 1515, 114]]
# INNER_RACE -> 1, OUTER_RACE -> 2, CAGE -> 3, BALL -> 4
FAULT_LABEL = [
    [2, 2, 2, 3, [1, 2]],
    [1, 2, 3, 2, 2],
    [2, [1, 2, 3, 4], 1, 1, 2]
]

MACHINING_PARAMETER_LABEL = [[35, 12], [37.5, 11], [40, 10]]

# 2^14 -> 32768
TIMESTEP = 32768


class dataSet(object):
    secondSamplingRatio = 16
    # TRAIN - > 1, TEST -> 0
    trainTestDist = [
        [1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0]
    ]
    # indicates alert 30 minutes ago
    alertTime = 30
    maxSampleTime = 300
    # cache configuration
    CACHE_DIR = "./cache"
    TRAIN_DATA_CACHE_NAME = "TRAIN_DATA"
    TEST_DATA_CACHE_NAME = "TEST_DATA"
    TRAIN_LABELS_CACHE_NAME = "TRAIN_LABELS"
    TEST_LABELS_CACHE_NAME = "TEST_LABELS"
    TRAIN_RUL_CACHE_NAME = "TRAIN_RUL"
    TEST_RUL_CACHE_NAME = "TEST_RUL"
    TRAIN_CONDITION_NAME = "TRAIN_CONDITION"
    TEST_CONDITION_NAME = "TEST_CONDITION"

    def __init__(self, secondSamplingRatio: int = 16, alertTime: int = 30):
        self.secondSamplingRatio = secondSamplingRatio
        self.alertTime = alertTime

    def get_condition_folder_path(self, conditionLabel: str):
        return os.path.join(XJTU_DATA_PATH, conditionLabel)

    def read_signal_file_to_array_by_path(self, path):
        pd_data = pd.read_csv(path)
        # print(pd_data)
        horizontal_signal = pd_data["Horizontal_vibration_signals"][::self.secondSamplingRatio].reshape(-1, 1)
        vertical_signal = pd_data["Horizontal_vibration_signals"][::self.secondSamplingRatio].reshape(-1, 1)
        np_data = np.concatenate((horizontal_signal, vertical_signal), axis=1)
        return np_data

    def read_reinforce_signal_file_to_array_by_path(self, path, ratio=8):
        pd_data = pd.read_csv(path)
        # print(pd_data)
        sample_base_index = np.arange(0,TIMESTEP,self.secondSamplingRatio)

        horizontal_signal = pd_data["Horizontal_vibration_signals"].reshape(-1, 1)
        vertical_signal = pd_data["Horizontal_vibration_signals"].reshape(-1, 1)
        np_data = np.concatenate((horizontal_signal, vertical_signal), axis=1)
        # reinforce data
        reinforce_step_range = self.secondSamplingRatio // ratio

        reinforce_dataset = []


        for step in range(0,reinforce_step_range):
            sample_index = sample_base_index + step
            reinforce_sample = np_data[sample_index,:]
            reinforce_dataset.append(reinforce_sample)

        return reinforce_dataset

    def get_all_data(self):
        train_data = []
        test_data = []
        train_alert_labels = []
        test_alert_labels = []
        train_rul_minutes = []
        test_rul_minutes = []
        train_condition_labels = []
        test_condition_labels = []

        for conditionIdx, conditionVal in enumerate([1, 2, 3]):
            print("# TOUCH condition folder :", CONDITION_DATA_FOLDER_LIST[conditionIdx])
            conditionFolderPath = os.path.join(XJTU_DATA_PATH, CONDITION_DATA_FOLDER_LIST[conditionIdx])
            for subSampleVal in range(1, 6):
                if (conditionVal == 1 and subSampleVal == 5) or (conditionVal == 3 and subSampleVal == 2):
                    print("Skip example condition",conditionVal,subSampleVal)
                    continue

                print("! TOUCH sample folder ", "Bearing%d_%d" % (conditionVal, subSampleVal))
                subSampleIdx = subSampleVal - 1
                bearingSampleFolderPath = os.path.join(conditionFolderPath,
                                                       "Bearing%d_%d" % (conditionVal, subSampleVal))
                isTrain = self.trainTestDist[conditionIdx][subSampleIdx]
                maxSampleIdx = CONDITION_MINUTES_NUMBER[conditionIdx][subSampleIdx]
                fault_label = FAULT_LABEL[conditionIdx][subSampleIdx]
                condition_label = MACHINING_PARAMETER_LABEL[conditionIdx]
                if isinstance(fault_label, int):
                    # one_hot_fault_label = [0 for _ in range(5)]
                    # one_hot_fault_label[fault_label] = 1
                    one_hot_fault_label = to_categorical(fault_label, num_classes=4)
                elif isinstance(fault_label, list):
                    raise TypeError("We skip the multi-label sample, it should not be list type")
                    one_hot_fault_label = [0 for _ in range(4)]
                    for i in fault_label:
                        one_hot_fault_label[i] = 1
                else:
                    raise ValueError("Inputed fault label does not fit type, given", type(fault_label),
                                     " required str or list")
                print("# MAX SAMPLE IDX :", maxSampleIdx)
                startMinute = 1
                endMinutes = maxSampleIdx
                if maxSampleIdx > self.maxSampleTime:
                    startMinute = endMinutes - self.maxSampleTime
                # add sample to data set
                for i in range(startMinute, endMinutes + 1):

                    print("@ SCANNING CSV ", conditionVal, subSampleVal, "%d.csv" % (i))
                    sampleCSVPath = os.path.join(bearingSampleFolderPath, "%d.csv" % (i))
                    np_data = self.read_signal_file_to_array_by_path(sampleCSVPath)

                    if isTrain:
                        # we need to reinforce train dataset
                        reinforce_data = self.read_reinforce_signal_file_to_array_by_path(sampleCSVPath,ratio=4)

                        if endMinutes - i <= self.alertTime:
                            # it's not normal
                            for reinforce_sample in reinforce_data:
                                train_data.append(reinforce_sample)
                                train_alert_labels.append(one_hot_fault_label)
                                train_rul_minutes.append(endMinutes - i)
                                train_condition_labels.append(condition_label)
                        else:
                            train_data.append(np_data)
                            train_alert_labels.append([1, 0, 0, 0])
                            train_rul_minutes.append(endMinutes - i)
                            train_condition_labels.append(condition_label)
                            # for reinforce_sample in reinforce_data:
                            #     train_data.append(reinforce_sample)
                            #     train_alert_labels.append([1, 0, 0, 0, 0])
                            #     train_rul_minutes.append(endMinutes - i)
                            #     train_condition_labels.append(condition_label)
                    else:
                        if endMinutes - i <= self.alertTime:
                            # it's not normal
                            test_data.append(np_data)
                            test_alert_labels.append(one_hot_fault_label)
                        else:
                            test_data.append(np_data)
                            test_alert_labels.append([1, 0, 0, 0])
                        test_condition_labels.append(condition_label)
                        test_rul_minutes.append(endMinutes - i)
                pass
        return np.array(train_data), np.array(train_alert_labels), np.array(train_rul_minutes), np.array(train_condition_labels), \
               np.array(test_data), np.array(test_alert_labels), np.array(test_rul_minutes), np.array(test_condition_labels)

    def get_all_cache_data(self):
        train_data_cache_path = os.path.join(self.CACHE_DIR, self.TRAIN_DATA_CACHE_NAME)
        train_alert_cache_path = os.path.join(self.CACHE_DIR, self.TRAIN_LABELS_CACHE_NAME)
        train_rul_cache_path = os.path.join(self.CACHE_DIR, self.TRAIN_RUL_CACHE_NAME)
        test_data_cache_path = os.path.join(self.CACHE_DIR, self.TEST_DATA_CACHE_NAME)
        test_alert_cache_path = os.path.join(self.CACHE_DIR, self.TEST_LABELS_CACHE_NAME)
        test_rul_cache_path = os.path.join(self.CACHE_DIR, self.TEST_RUL_CACHE_NAME)
        train_condition_cache_path = os.path.join(self.CACHE_DIR,self.TRAIN_CONDITION_NAME)
        test_condition_cache_path = os.path.join(self.CACHE_DIR, self.TEST_CONDITION_NAME)
        if os.path.exists(train_data_cache_path + ".npy"):
            print("! Using cached data")
            return np.load(train_data_cache_path + ".npy"), np.load(train_alert_cache_path + ".npy"), np.load(
                train_rul_cache_path + ".npy"), np.load(train_condition_cache_path+".npy"), \
                   np.load(test_data_cache_path + ".npy"), np.load(test_alert_cache_path + ".npy"), np.load(
                test_rul_cache_path + ".npy"), np.load(test_condition_cache_path+".npy"),
        else:
            print("! Scanning directory data")
            train_data, train_alert_labels, train_rul_minutes, train_condition, test_data, test_alert_labels, test_rul_minutes, test_condition = self.get_all_data()
            np.save(train_data_cache_path, train_data)
            np.save(train_alert_cache_path, train_alert_labels)
            np.save(train_rul_cache_path, train_rul_minutes)
            np.save(test_data_cache_path, test_data)
            np.save(test_alert_cache_path, test_alert_labels)
            np.save(test_rul_cache_path, test_rul_minutes)
            np.save(train_condition_cache_path, train_condition)
            np.save(test_condition_cache_path, test_condition)
            return train_data, train_alert_labels, train_rul_minutes, train_condition, test_data, test_alert_labels, test_rul_minutes, test_condition


if __name__ == "__main__":

    data = dataSet()
    train_data, train_alert_labels, train_rul_minutes, train_condition, test_data, test_alert_labels, test_rul_minutes, test_condition = data.get_all_cache_data()
    print("Train and test shape", train_data.shape, test_data.shape, train_rul_minutes.shape, train_alert_labels.shape, train_condition.shape)
    # data.get_all_data()
    # np_data = data.read_signal_file_to_array_by_path(
    #     r"E:\IDM_download\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_1\1.csv")
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Qt5Agg")

    idList = [0,50,75,100,150]
    for idx,sampleIdx in enumerate(idList):
        plt.subplot("%d%d%d"%(len(idList),1,idx+1))
        plt.plot(train_data[sampleIdx,:,0],label="Horizon_%d"%(sampleIdx))
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.legend()
    plt.show()




    plt.plot(np.argmax(train_alert_labels,axis=1), label="TRAIN_ALERT_LABEL")
    plt.plot(train_rul_minutes,label="Train RUL")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()
