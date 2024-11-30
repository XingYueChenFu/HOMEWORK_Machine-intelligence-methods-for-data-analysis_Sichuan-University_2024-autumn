# EZModel

获取数据方法
```py
import tensorflow as tf  
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
然后可以将`mnist.npz`

从`C:\Users\<YourUsername>\.keras\datasets\`中将`mnist.npz`复制出来
放到`EZModel.ipynb`所在文件夹里的`data`中
