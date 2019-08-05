from keras.layers import *
from keras.models import Model,Input
from keras.optimizers import Adam,SGD
from keras.regularizers import l2

def residual_block(x, filters: tuple, changeDim: bool, kernel_size=3, pooling_size=1, dropout=0.5):
    k1, k2 = filters
    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv1D(k1, kernel_size, strides=1, padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    if changeDim:
        out = Conv1D(k2, kernel_size, strides=2, padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(out)
        pooling = Conv1D(k2, kernel_size, strides=2, padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(x)
    else:
        out = Conv1D(k2, kernel_size, strides=1, padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(out)
        pooling = x

    # out = merge([out,pooling],mode='sum')
    out = add([out, pooling])
    return out


def build_residual_rcnn(time_length, input_channel, output_class_num, block_depth, dropout=0.5):
    inp = Input(shape=(time_length, input_channel), name="signal_input")


    out = Conv1D(16, 7,strides=1,padding="same",kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(inp)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    BASE_DIM = 16
    BLOCK_PER_NUM = 4
    for curDepth in range(block_depth):
        curBlockDepth = curDepth // BLOCK_PER_NUM
        curChannel = BASE_DIM * (2 ** curBlockDepth)
        if curDepth % BLOCK_PER_NUM != 0 or curDepth == 0:
            out = residual_block(out, (curChannel, curChannel), False, dropout=dropout)
        else:
            out = residual_block(out, (curChannel, curChannel), True, dropout=dropout)

    # add flatten
    out = Flatten()(out)
    condition_inp = Input(shape=(2,), name="condition_input")

    out_class = Dense(output_class_num,activation="softmax",name="dense_class_5")(out)
    out_rul = Dense(1,name="dense_rul")(out)

    model = Model([inp,condition_inp],[out_class,out_rul])
    adam = Adam(lr=0.01)
    model.compile(adam,
                  loss={
                      "dense_class_5":"categorical_crossentropy",
                      "dense_rul":"logcosh"
                  },
                  loss_weights={
                      "dense_class_5": 1 ,
                      "dense_rul": 1
                  },
                  metrics={
                      "dense_class_5": ["acc"],
                      "dense_rul": ["mse","mae"]
                  })
    return model

def build_residual_rcnn_for_prognostic(time_length, input_channel, output_class_num, block_depth, dropout=0.5):
    inp = Input(shape=(time_length, input_channel), name="signal_input")
    condition_inp = Input(shape=(2,), name="condition_input")
    condition_embedding = RepeatVector(time_length)((Dense(2,name="condition_embedding")(condition_inp)))

    out = multiply([inp,condition_embedding])

    out = Conv1D(32, 5,strides=1,padding="same")(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    BASE_DIM = 32
    BLOCK_PER_NUM = 4
    for curDepth in range(block_depth):
        curBlockDepth = curDepth // BLOCK_PER_NUM
        curChannel = BASE_DIM * (2 ** curBlockDepth)
        if curDepth % BLOCK_PER_NUM != 0 or curDepth == 0:
            out = residual_block(out, (curChannel, curChannel), False, dropout=dropout)
        else:
            out = residual_block(out, (curChannel, curChannel), True, dropout=dropout)

    # add flatten
    out = Flatten()(out)
    # out = Dense(128,)(out)

    # condition_embedding = Flatten()(RepeatVector(32768//4)(Flatten()((Embedding(2,2,name="condition_embedding")(condition_inp)))))
    # out = multiply([out,condition_embedding])
    # out = Flatten()(out)

    out_class = Dense(output_class_num,activation="softmax",name="dense_class_4")(out)
    # out_rul = Dense(1,name="dense_rul")(out)

    model = Model([inp,condition_inp],out_class)
    # model.compile(adam,
    #               loss={
    #                   "dense_class_5":"categorical_crossentropy",
    #                   "dense_rul":"logcosh"
    #               },
    #               loss_weights={
    #                   "dense_class_5": 1 ,
    #                   "dense_rul": 1
    #               },
    #               metrics={
    #                   "dense_class_5": ["acc"],
    #                   "dense_rul": ["mse","mae"]
    #               })
    adam = Adam(lr=0.000001)
    sgd = SGD(lr=0.001,momentum=0.9)
    model.compile(adam,loss="categorical_crossentropy",metrics=["acc"])
    return model

if __name__ == "__main__":
    model = build_residual_rcnn_for_prognostic(2048,2,5,20)
    print(model.summary())
