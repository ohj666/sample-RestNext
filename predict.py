import tensorflow as tf
from tensorflow.keras import datasets
import tensorflow as tf
from resnext_block import build_ResNeXt_block
import cv2
import numpy as np
# (x_trian,y_trian),(x_test,y_test)=datasets.cifar10.load_data()
NUM_CLASSES=5
img=r"D:\data\over_pokeman\bulbasaur"
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


model=ResNeXt50()
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            #准确率（概率分布）

            metrics=['sparse_categorical_accuracy'])
model.build(input_shape=(1168,224,224,3))
model.load_weights("resnext.h5")
model.summary()


# model=tf.keras.models.load_model("yang.h5")
# model.summary()
img=cv2.imread(r"pikachu2.jpeg")
cv2.imshow('img',img)
#输出数据集中的一张照片
# img=x_trian[250]
img=cv2.resize(img,(224,224))
img=img.reshape(1,224,224,3)
img=tf.cast(img,tf.float32)
img=(img/255.0)*2-1


result=np.argmax(model.predict(img))
cv2.waitKey(0)
if result==0:
    print("this is bulbasaur")
if result==1:
    print("this is charmander")
if result==2:
    print("this is mewtwo")
if result==3:
    print("this is pikachu")
if result==4:
    print("this is squirtle")

