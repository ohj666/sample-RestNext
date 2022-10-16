import tensorflow as tf
from resnext_block import build_ResNeXt_block
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot as plt
NUM_CLASSES = 5
gpus=tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    gpu0=gpus[0]
    tf.config.experimental.set_memory_growth(gpu0,True)
    tf.config.set_visible_devices([gpu0],"GPU")

class ResNeXt(tf.keras.Model):
    def __init__(self, repeat_num_list, cardinality):
        if len(repeat_num_list) != 4:
            raise ValueError("The length of repeat_num_list must be four.")
        super(ResNeXt, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        self.block1 = build_ResNeXt_block(filters=128,
                                          strides=1,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[0])
        self.block2 = build_ResNeXt_block(filters=256,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[1])
        self.block3 = build_ResNeXt_block(filters=512,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[2])
        self.block4 = build_ResNeXt_block(filters=1024,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[3])
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)

        return x


def ResNeXt50():
    return ResNeXt(repeat_num_list=[3, 4, 6, 3],
                   cardinality=32)


def ResNeXt101():
    return ResNeXt(repeat_num_list=[3, 4, 23, 3],
                   cardinality=32)
from tensorflow.keras import datasets,Sequential
x=np.load("pokeman_x.npy")
y=np.load("pokeman_y.npy")
x_train,x_test,y_train,y_test=train_test_split(x,y)
tf.cast(y_train,tf.float32)
tf.cast(y_test,tf.float32)
x_train=(x_train/255.)*2-1
x_test=(x_test/255.)*2-1
model=ResNeXt50()
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            #准确率（概率分布）
            metrics=['sparse_categorical_accuracy'])
mymodel=model.fit(x_train,y_train,batch_size=64,epochs=30,validation_data=(x_test,y_test))
model.summary()
model.save_weights("resnext.h5")
#绘制损失图和准确率的图

plt.figure(figsize=(16, 9))
plt.suptitle('loss and accuracy', fontsize=14, fontweight="bold")
#训练集损失率
plt.subplot(2,2,1)
plt.title(" train Loss ")
plt.xlabel('Epoch')
plt.ylabel('loss1')
plt.plot(mymodel.history['loss'])
#测试集损失率

plt.subplot(2,2,2)
plt.title(" text Loss ")
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(mymodel.history['val_loss'])
#测试集准确率

plt.subplot(2,2,3)
plt.title("train accuracy")
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.plot(mymodel.history['sparse_categorical_accuracy'])

#测试集准确率

plt.subplot(2,2,4)
plt.title("text accuracy")
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.plot(mymodel.history['val_sparse_categorical_accuracy'])
plt.show()