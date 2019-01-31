
from collections import namedtuple
from time import time
import tensorflow as tf

from Globals import SAMPLE_LENGTH, BATCH_SIZE

ModelInfo = namedtuple('ModelInfo', 
                       'data, labels, loss, optimizer, output, allOutputs, finalState')
ModelInfo.__new__.__defaults__ = (None,) * len(ModelInfo._fields)


class RnnLearner:
    def __init__(self):
        tf.reset_default_graph()

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
        
    def BuildModel(self):
        data = tf.placeholder(dtype=tf.float32, 
                              shape=(None,SAMPLE_LENGTH, 1), 
                              name='data')
    
        lstmCell = tf.contrib.rnn.LSTMBlockCell(num_units=16)

        rawOutputs, final_state = tf.nn.dynamic_rnn(
                lstmCell, 
                data, 
                dtype=tf.float32)
        
        outputs = rawOutputs[:, -1, :]   # wtf, dudes
        output = tf.layers.dense(outputs, 1, kernel_initializer=tf.orthogonal_initializer())

        labels = tf.placeholder(dtype=tf.float32, shape=(None,1), name="labels")

        loss = tf.losses.mean_squared_error(labels, output)
        optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(loss)

        result = ModelInfo(data=data,
                           labels=labels,
                           loss=loss,
                           optimizer=optimizer,
                           output=output,
                           allOutputs=outputs,
                           finalState=final_state)
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
        