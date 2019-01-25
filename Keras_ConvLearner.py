
from time import time
from pathlib import Path
import keras
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf

from Globals import RESHAPED_DIMS, BATCH_SIZE

def _calculateLoss(yTrue, yPred):
    return tf.reduce_sum(tf.multiply(tf.abs(tf.subtract(yTrue, yPred)), 4e3))

class KerasConvLearner:
    def __init__(self):
        self.learning_rate = 1e-5
        self.model = self.BuildModel()
        self.tensorboard = self.InitStatsWriter(self.model)
        self.SaveWeightsFilename = "ConvLearnerWeights"
        self.LoadModelInfo()
        print(self.model.summary())

    def InitStatsWriter(self, model):        
        result = keras.callbacks.TensorBoard(log_dir=f"tensorboard/{int(time())}",
                                             write_graph=False,
                                             update_freq=10*BATCH_SIZE)
        result.set_model(model)
        return result

    def BuildModel(self):
        kernelInit = keras.initializers.VarianceScaling(2.0)
    
        data = keras.layers.Input(shape=RESHAPED_DIMS, name='data')
    
        conv_1 = Conv2D(filters=32, 
                        kernel_size=8, 
                        strides=2, 
                        input_shape=RESHAPED_DIMS, 
                        activation='relu',
                        kernel_initializer=kernelInit,
                        name='conv_1'
                        )(data)
                
        conv_2 = Conv2D(filters=64,
                        kernel_size=4,
                        strides=2, 
                        activation='relu',
                        kernel_initializer=kernelInit,
                        name='conv_2'
                        )(conv_1)
                
        conv_3 = Conv2D(filters=64, 
                        kernel_size=4, 
                        strides=2, 
                        activation='relu',
                        kernel_initializer=kernelInit,
                        name='conv_3'
                        )(conv_2)

        flattened = Flatten()(conv_3)
        normalized = BatchNormalization(name='normalized')(flattened)

        fc_1 = Dense(units = 256, 
                     activation='relu',
                     kernel_initializer=kernelInit,
                     name='fc_1')(normalized)
        
        output = Dense(units=1,
                       activation='relu',
                       kernel_initializer=kernelInit,
                       name='output')(fc_1)

        result = keras.models.Model(inputs=data, outputs=output)
        optimizer = Adam(lr=self.learning_rate)
        result.compile(optimizer, loss=_calculateLoss)
        return result

    def Fit(self, iBatch, iLabels):
        self.model.fit(
                x = iBatch, 
                y = iLabels,
                epochs=1, 
                batch_size=len(iLabels), 
                verbose=0,
                callbacks=[self.tensorboard])

    def LoadModelInfo(self):
        weightsFile = Path(self.SaveWeightsFilename)
        if(weightsFile.is_file()):
            self.model.load_weights(self.SaveWeightsFilename)
            print("*** Model Weights Loaded ***")

    def SaveModelInfo(self):
        self.model.save_weights(self.SaveWeightsFilename)


        