#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import pickle as pkl
from random import shuffle
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-f','--file_name',type=str,required=True)
    parser.add_argument('-w','--window_size',type=int, default=20,required=False)
    parser.add_argument('-i','--input_dimension',type=int, default=8,required=False)
    parser.add_argument('-iter','--iteration', type=int, default=5001,required=False)
    parser.add_argument('-n','--neurons', type=int, default=100,required=False)
    return parser.parse_args()

def main(args):
    n_inputs=args.input_dimension
    n_windows=args.window_size
    n_outputs=2 #  0 for normal, 1 for anormal
    n_neurons=100
    learning_rate=0.001
    iteration=args.iteration

    train_input, train_output, test_input, test_output = read_data(args.file_name)
    ori_train_len=len(train_input)
    ori_test_len=len(test_input)

    rest= ori_train_len%n_windows
    if rest:
        X_batches=np.append(train_input, np.zeros((n_windows-rest,n_inputs)))
        Y_batches=np.append(train_output, np.zeros((n_windows-rest,n_outputs)))
    rest=ori_test_len%(n_windows)
    if rest:
        X_test=np.append(test_input,np.zeros((n_windows-rest,n_inputs)))
        Y_test=np.append(test_output,np.zeros((n_windows-rest,n_outputs)))
    X_batches=X_batches.reshape(-1,n_windows,n_inputs)
    Y_batches=Y_batches.reshape(-1,n_windows,n_outputs)
    X_test=X_test.reshape(-1,n_windows,n_inputs)
    Y_test=Y_test.reshape(-1,n_windows,n_outputs)

    tf.reset_default_graph()

    X=tf.placeholder(tf.float32, [None, n_windows, n_inputs])
    y=tf.placeholder(tf.float32, [None, n_windows, n_outputs])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    rnn_outputs, states= tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32) 
    
    stacked_rnn_ouput= tf.reshape(rnn_outputs, [-1,n_neurons])
    stacked_outputs=tf.layers.dense(stacked_rnn_ouput, n_outputs)
    outputs=tf.reshape(stacked_outputs,[-1,n_windows,n_outputs])

    loss = tf.reduce_sum(tf.square(outputs-y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        for iters in range(iteration):
            sess.run(training_op,feed_dict={X:X_batches, y: Y_batches})
            if iters % 200==0:
                mse=loss.eval(feed_dict={X:X_batches, y:Y_batches})
                y_pred=sess.run(outputs,feed_dict={X:X_test})
                y_pred=y_pred.reshape(-1,2)
                prediction=np.argmax(y_pred,axis=1)
                correct = np.argmax(test_output,axis=1)
                tp=fp=fn=tn=0
                for i in range(ori_test_len):
                    if prediction[i]:
                        if correct[i]:
                            tp+=1.0
                        else:
                            fp+=1.0
                    elif correct[i]:
                        fn+=1.0
                    else:
                        tn+=1.0
                precs=tp/(tp+fp)*100
                recc=tp/(tp+fn)*100
                accu=(tp+tn)/(tp+tn+fp+fn)*100
                mr=fn/(tp+fn)*100
                far=fp/(tn+fp)*100
                print("%d\tF1:%f\taccu:%f\tpr:%f\trec:%f\tmr:%f\tfar:%f"
                %(iters,2*precs*recc/(precs+recc),accu,precs,recc,mr,far))

def read_data(file_name):
    with open(file_name,'r') as f:
        flow_data=pkl.load(f)
    normal_flow=[]
    anormal_flow=[]
    sum_=np.zeros(6)
    num_=np.zeros(6)
    for flow in flow_data:
        if flow[0]:
            anormal_flow.append([np.array(flow[2:]), np.array([0,1])])
        else:
            normal_flow.append([np.array(flow[2:]), np.array([1,0])])
        for i in range(4,10):
            if flow[i]!=None:
                num_[i-4]+=1
                sum_[i-4]+=flow[i]
    avg_=np.divide(sum_,num_)
    print("%d normal flow and %d anormal flow readed"%(len(normal_flow),len(anormal_flow)))
    #shuffle(normal_flow)
    #shuffle(anormal_flow)
    for flow in normal_flow+anormal_flow:
        for i in range(2,8):
            if flow[0][i]==None:
                flow[0][i]=0#avg_[i-2]
    #normal_flow=normal_flow[:-len(normal_flow)/5]
    train_data=normal_flow[:-len(normal_flow)/10]+anormal_flow[:-len(anormal_flow)/10]
    test_data=normal_flow[-len(normal_flow)/10:]+anormal_flow[-len(anormal_flow)/10:]
    #shuffle(train_data)
    #shuffle(test_data)
    train_input=[]
    train_output=[]
    test_input=[]
    test_output=[]
    for flow in train_data:
        train_input.append(flow[0])
        train_output.append(flow[1])
    for flow in test_data:
        test_input.append(flow[0])
        test_output.append(flow[1])
    train_input=np.array(train_input)
    train_output=np.array(train_output)
    test_input=np.array(test_input)
    test_output=np.array(test_output)
    return train_input,train_output,test_input,test_output

if __name__=='__main__':
    args=parse_args()
    main(args)
