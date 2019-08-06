from dataset import dataSet
from models import *
from keras.callbacks import *
import matplotlib.pyplot as plt
import matplotlib
import itertools
import tensorflow as tf
import os.path


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

matplotlib.use("Qt5Agg")

data = dataSet()
train_data, train_alert_labels, train_rul_minutes, train_condition, test_data, test_alert_labels, test_rul_minutes, test_condition = data.get_all_cache_data()
train_rul_minutes = train_rul_minutes.reshape(-1, 1)
test_rul_minutes = test_rul_minutes.reshape(-1,1)

PREDICT = False

TOTAL_EPOCH = 10000

BATCH_SIZE = 64

TOTAL_BATCH_NUM = train_alert_labels.shape()

for depth in [25, 20, 15, 10]:
    log_dir = "rcnn_logs/"
    dropout = 0.1
    train_name = 'Resnet_block_%s_embedding_%s' % (depth, dropout)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (train_name)
    MODEL_NAME = '%s.kerasmodel' % (train_name)
    model_class, model_rul = build_residual_shared_weights(2048, 2, 4, depth, dropout=dropout)


    # Tensorboard callback compatible setting
    log_path = os.path.join(log_dir,train_name)
    class_callback = TensorBoard(log_path+"_class_4")
    class_callback.set_model(model_class)
    class_metric_train_name = ["loss","acc"]
    class_metric_val_name = ["val_loss", "val_acc"]


    rul_callback = TensorBoard(log_path+"_rul_regr")
    rul_callback.set_model(model_rul)
    rul_metric_train_name = ["loss","mse","mae"]
    rul_metric_val_name = ["val_loss", "val_mse", "val_mae"]

    if not PREDICT:
        for epoch in range(TOTAL_EPOCH):
            # segment the sample

            class_logs = model_class.train_on_batch([train_data,train_condition],train_alert_labels)
            write_log(class_callback,class_metric_train_name,class_logs, epoch)
            rul_logs = model_rul.train_on_batch([train_data,train_condition],train_rul_minutes)
            write_log(class_callback,class_metric_train_name, class_logs, epoch)




        tb_cb = TensorBoard(log_dir=log_dir + train_name)
        ckp_cb = ModelCheckpoint(MODEL_CHECK_PT, monitor='val_loss', save_weights_only=True, verbose=1,
                                 save_best_only=True, period=5)
        import os.path

        if os.path.exists(MODEL_CHECK_PT):
            print("Load weights successfully")
            model.load_weights(MODEL_CHECK_PT)

        print('Model has been established.')

        model.fit([train_data,train_condition], [train_alert_labels, train_rul_minutes],
                  batch_size=16, epochs=10000,
                  callbacks=[tb_cb, ckp_cb],
                  initial_epoch=0,
                  validation_data=([test_data,test_condition], [test_alert_labels, test_rul_minutes]))

        model.save(MODEL_NAME)
    else:
        if os.path.exists(MODEL_CHECK_PT):
            print("Load weights successfully : %s " % (MODEL_CHECK_PT))
            model.load_weights(MODEL_CHECK_PT)
        else:
            raise FileExistsError("No weights found : %s, please train it first" % (MODEL_CHECK_PT))

        test_alert_labels_pred, test_rul_minutes_pred = model.predict([test_data,test_condition])
        print(model.evaluate([test_data,test_condition],[test_alert_labels,test_rul_minutes]))
        plt.figure()
        plt.plot(np.argmax(test_alert_labels,axis=1),label="TEST ALERT")
        plt.plot(np.argmax(test_alert_labels_pred,axis=1),'--',label="PRED ALERT")
        plt.legend()
        plt.show()
        plt.close()

        train_alert_labels_pred, train_rul_minutes_pred = model.predict([train_data, train_condition])
        print(model.evaluate([test_data, test_condition], [test_alert_labels, test_rul_minutes]))
        plt.figure()
        plt.plot(np.argmax(train_alert_labels, axis=1), label="TRAIN ALERT")
        plt.plot(np.argmax(train_alert_labels_pred, axis=1),'--', label="PRED ALERT")
        plt.legend()
        plt.show()
        plt.close()
        from sklearn.metrics import confusion_matrix

        print(test_alert_labels_pred.shape, test_alert_labels.shape)
        classesTextList = ["Normal", "Inner race", "Outer race", "Cage"]
        confusionMatrix = confusion_matrix(np.argmax(test_alert_labels, axis=1),
                                           np.argmax(test_alert_labels_pred, axis=1))
        print(confusionMatrix)
        plt.figure()

        plot_confusion_matrix(confusionMatrix, classes=classesTextList, normalize=True,
                              title="Normalized Confusion Matrix")
        plt.show()
        plt.close()

        fig = plt.figure()
        plt.plot(test_rul_minutes_pred, label="Predicted RUL")
        plt.plot(test_rul_minutes, label="Real RUL")

        plt.ylabel("Remaining Useful Life (minutes)")
        plt.xlabel("Sample")
        plt.legend()
        plt.show()
