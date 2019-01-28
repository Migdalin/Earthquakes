
from collections import namedtuple
from time import time
import tensorflow as tf

from Globals import RESHAPED_DIMS

ModelInfo = namedtuple('ModelInfo', 
                       'data, labels, loss, optimizer, output')
ModelInfo.__new__.__defaults__ = (None,) * len(ModelInfo._fields)

ConvArgs = namedtuple('ConvArgs',
                      'layerInput, numFilters, filterSize, stride, init, namePrefix')
ConvArgs.__new__.__defaults__ = (None,) * len(ConvArgs._fields)

class ConvLearner:
    def __init__(self):
        self.session = tf.Session()
        self.learning_rate = 1e-5
        self.total_step_count = 0
        self.model = self.BuildModel()
        self.saver = tf.train.Saver()
        self.InitStatsWriter()
        self.session.run(tf.global_variables_initializer())

    def InitStatsWriter(self):        
        self.statsWriter = tf.summary.FileWriter(f"tensorboard/{int(time())}")
        tf.summary.scalar("Loss", self.model.loss)
        self.writeStatsOp = tf.summary.merge_all()
        self.SaveWeightsFilename = "./checkpoint/ConvLearnerCheckpoint.ckpt"

    def WriteStats(self, feedDict):
        summary = self.session.run(self.writeStatsOp, feed_dict=feedDict)
        self.statsWriter.add_summary(summary, self.total_step_count)
        
    def BuildConv2D(self, convArgs):
        channelAxis = 3
        filterShape = [convArgs.filterSize, 
                       convArgs.filterSize, 
                       convArgs.layerInput.get_shape()[channelAxis], 
                       convArgs.numFilters]

        filters = tf.get_variable(shape=filterShape, 
                                  dtype=tf.float32,
                                  initializer=convArgs.init,
                                  name=convArgs.namePrefix + 'filters')
        
        conv = tf.nn.conv2d(input=convArgs.layerInput,
                            filter=filters, 
                            strides=[1,convArgs.stride,convArgs.stride,1], 
                            padding='VALID')

        activated = tf.nn.relu(conv)
        return activated

    def BuildModel(self):
        kernelInit = tf.contrib.layers.xavier_initializer()
    
        data = tf.placeholder(dtype=tf.float32, 
                              shape=(None,) + RESHAPED_DIMS, 
                              name='data')
    
        conv_1 = self.BuildConv2D(
                ConvArgs(layerInput = data, 
                         numFilters = 32,
                         filterSize = 8,
                         stride = 2,
                         init = kernelInit,
                         namePrefix='c1'))

        conv_2 = self.BuildConv2D(
                ConvArgs(layerInput = conv_1, 
                         numFilters = 64,
                         filterSize = 4,
                         stride = 2,
                         init = kernelInit,
                         namePrefix='c2'))

        conv_3 = self.BuildConv2D(
                ConvArgs(layerInput = conv_2, 
                         numFilters = 128,
                         filterSize = 4,
                         stride = 2,
                         init = kernelInit,
                         namePrefix='c3'))

        flattened = tf.layers.Flatten()(conv_3)
        fc_1 = tf.layers.Dense(units = 256, 
                               activation='relu',
                               kernel_initializer=kernelInit,
                               name='fc_1')(flattened)
        
        output = tf.layers.Dense(units=1,
                                 kernel_initializer=kernelInit,
                                 name='output')(fc_1)

        labels = tf.placeholder(dtype=tf.float32, 
                               shape=(None,1),
                               name="labels")

        loss = tf.reduce_mean(tf.multiply(tf.abs(tf.subtract(labels, output)), 4e3))
        optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(loss)

        result = ModelInfo(data=data,
                           labels=labels,
                           loss=loss,
                           optimizer=optimizer,
                           output=output)
        return result        

    def Fit(self, iBatch, iLabels):
        feed_dict={self.model.data: iBatch,
                   self.model.labels: iLabels}
        self.session.run(self.model.optimizer, feed_dict)
    
        self.total_step_count += 1
        if((self.total_step_count % 10) == 0):
            self.WriteStats(feed_dict)

    def Predict(self, iData):
        feedDict={self.model.data: iData}
        result = self.session.run(self.model.output, feed_dict=feedDict)
        return result.flatten()

    def Save(self):
        self.saver.save(self.session, self.SaveWeightsFilename)

    def Load(self):
        if(tf.train.checkpoint_exists(self.SaveWeightsFilename)):
            self.saver.restore(self.session, self.SaveWeightsFilename)
        