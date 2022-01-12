# AnomalyDetection with INT

Please refer next link in ONOS Wiki about way to build virtual network testbed with INT.

https://wiki.onosproject.org/display/ONOS/In-band+Network+Telemetry+%28INT%29+with+ONOS+and+P4


***
* DataStreamProcessor.py

Read Flow data from Influx db and save it as pkl. If you want to save with list, you should use --save_as_list option. (for RNN, you should save as list)

* RNN.py 

RNN using Tensorflow. You should put the flow data file name. It will show you the f1 score at every 200 iteration.


* flow_data_list.pkl 

My flow data. It contains 43584 normal data and 1341 abnormal data.

***

Running example) ./rnn.py -f flow_data_list.pkl
