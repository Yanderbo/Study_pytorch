<h1>class_two</h1>
<h2>NN模型识别结果</h2>
<p>Got 56077 / 60000 with accuracy 93.46</br>
Got 9329 / 10000 with accuracy 93.29</p>
<h2>CNN模型识别结果</h2>
<p>Got 57810 / 60000 with accuracy 96.35</br>
Got 9667 / 10000 with accuracy 96.67</p>
<h2>RNN模型</h2>
<p>一般用于文字识别（语音），一般不用于图像识别</p>
<p>Got 57537 / 60000 with accuracy 95.89</br>
Got 9609 / 10000 with accuracy 96.09</p>
<p>将RNN改成GRU准确率为：</br>
Got 59439 / 60000 with accuracy 99.06</br>
Got 9879 / 10000 with accuracy 98.79</p>
<i>也可将RNN改为LSTM</i></r>
<img src="lstm_c0.png">
<h2>BRNN:双向</h2>
<i>链接输入乘2</i>
<img src="BRNN.jpg">
<h2>保存，下载状态</h2>
<img src="checkpoint.png">
<img src="checkpoint1.png">