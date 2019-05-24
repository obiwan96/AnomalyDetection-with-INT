#!/usr/bin/env python
"""Based on 'Tutorial on using the InfluxDB client.'"""
# -*- coding: utf-8 -*-

################################################################################
# DataStreamProcessor
# Save Influx db as pickle. 
# The result is a list of dictionary with next keys.
# If you use the save_as_list option, it will return list with left index.
# Also, if you use the save_as_list option, duration will be turned to int 
# with unit as sencond
#
#   Flow_id : string with form of 'source_ip>dest_ip'. ex)'10.0.0.1>10.0.0.2'
#   Start_time : datetime.datetime object
# 0 Anomaly : 0 or 1, int
# 1 Path : List of string. Path could be []. ex)['11:22:12', '11:21:12']
# 2 Protocol : int. 17 for UDP.
# 3 Duration : datetime.timedelta object. max value could be given by datetime.timedelta(0,args.max_duration, 0)
# 4 Hop_latency : None if not observed, else int.
# 5 Egress_time : None if not observed, else int.
# 6 Flow_latency : None if not observed, else int.
# 7 Sink_time : None if not observed, else int.
# 8 Port_tx_utilize : None if not observed, else int.
# 9 Queue_occupancy : None if not observed, else int.
#
################################################################################

import argparse
import json
from influxdb import InfluxDBClient
import unicodedata
from datetime import datetime, timedelta
import pickle as pkl
from os import listdir

def main(args, host='localhost', port=8086):
    """Instantiate a connection to the InfluxDB."""
    user = 'root'
    password = 'root'
    dbname = 'INTdatabase'
    #dbuser = 'dpnm'
    #dbuser_password = 'dpnm'
    query = 'select * from '
    max_duration=timedelta(0,args.max_duration)

    client = InfluxDBClient(host, port, user, password, dbname)
            
    #print("Switch user: " + dbuser)
    #client.switch_user(dbuser, dbuser_password)

    print("Querying measurements name")
    result= client.query('show measurements')
    measurements=list(result.get_points())
    measurements=[unicodedata.normalize('NFKD',measurement.values()[0]).encode('ascii',
        'ignore')for measurement in measurements]
    print("get %d measurements\n"%len(measurements))
    
    if args.print_query:
        for measurement in measurements:
            result = client.query(query+'\"'+measurement+'\"')
            key=list(result.keys())
            value=list(result.get_points())
            print("key : {0}".format(key[0]))
            print("value : {0}\n".format(value[0]))

    print("Querying data")
    FlowData=[]
    if 'flow_stat' in measurements:
        result = client.query(query+'\"flow_stat\"')
        values=list(result.get_points())
        for value in values:
            date_time_str=value['time'].encode('utf-8')
            date_time_obj=datetime.strptime(date_time_str[:-4],
                '%Y-%m-%dT%H:%M:%S.%f')
            already_in_data=False
            for data in FlowData:
                if (data['Flow_id']==value['flow_id'].encode('utf-8') and
                    data['Protocol'] == value['proto'] and
                    date_time_obj - data['Start_time'] < max_duration ):
                    #same flow already in flow data
                    already_in_data=True
                    if date_time_obj - data['Start_time'] > data['Duration']:
                        data['Duration']=date_time_obj - data['Start_time']
                    data['Flow_latency'].append(value['flow_latency'])
                    if not value['path'].encode('utf-8') in data['Path']:
                        data['Path'].append(value['path'].encode('utf-8'))
                    data['Sink_time'].append(value['sink_time'])
                    break
            if not already_in_data:
                data={'Flow_id':value['flow_id'].encode('utf-8'), \
                'Start_time' : date_time_obj, \
                'Anomaly' : 1 if args.anomaly else 0, \
                'Path' : [value['path'].encode('utf-8')], 'Protocol' : value['proto'], \
                'Duration' : timedelta(0), 'Hop_latency' : [], \
                'Egress_time' : [], 'Flow_latency':[], \
                'Sink_time' : [value['sink_time']], \
                'Port_tx_utilize' : [], 'Queue_occupancy' : []}
                FlowData.append(data)
    if 'flow_hop_latency' in measurements:
        result = client.query(query+'\"flow_hop_latency\"')
        values=list(result.get_points())
        for value in values[:]:
            date_time_str=value['time'].encode('utf-8')
            date_time_obj=datetime.strptime(date_time_str[:-4],
                '%Y-%m-%dT%H:%M:%S.%f')
            already_in_data=False
            for data in FlowData:
                if (data['Flow_id']==value['flow_id'].encode('utf-8') and
                    data['Protocol'] == value['proto'] and
                    date_time_obj - data['Start_time'] < max_duration ):
                    #same flow already in flow data
                    already_in_data=True
                    if date_time_obj - data['Start_time'] > data['Duration']:
                        data['Duration']=date_time_obj - data['Start_time']
                    data['Hop_latency'].append(value['value'])
                    data['Egress_time'].append(value['egress_time'])
                    break
            if not already_in_data:
                data={'Flow_id':value['flow_id'].encode('utf-8'), \
                'Start_time' : date_time_obj, \
                'Anomaly' : 1 if args.anomaly else 0, \
                'Path' : [], 'Protocol' : value['proto'], \
                'Duration' : timedelta(0), 'Hop_latency' : [value['value']], \
                'Egress_time' : [value['egress_time']], 'Flow_latency':[], \
                'Sink_time' : [], 'Port_tx_utilize' : [], 'Queue_occupancy' : []}
                FlowData.append(data)
    if 'port_tx_utilize' in measurements:
        result = client.query(query+'\"port_tx_utilize\"')
        values=list(result.get_points())
        for value in values:
            date_time_str=value['time'].encode('utf-8')
            date_time_obj=datetime.strptime(date_time_str[:-4],
                '%Y-%m-%dT%H:%M:%S.%f')
            for data in FlowData:
                for path in data['Path']:
                    if (value['sw_id'] in [int(sw_id) for sw_id in path.split(':')] and 
                        date_time_obj - data['Start_time'] < data['Duration'] ):
                        data['Port_tx_utilize'].append(value['value'])
                        break
    if 'queue_occupancy' in measurements:
        result = client.query(query+'\"queue_occupancy\"')
        values=list(result.get_points())
        for value in values:
            date_time_str=value['time'].encode('utf-8')
            date_time_obj=datetime.strptime(date_time_str[:-4],
                '%Y-%m-%dT%H:%M:%S.%f')
            for data in FlowData:
                for path in data['Path']:
                    if (value['sw_id'] in [int(sw_id) for sw_id in path.split(':')] and 
                        date_time_obj - data['Start_time'] < data['Duration'] ):
                        data['Queue_occupancy'].append(value['value'])
                        break
    print('Find %d flows\n'%len(FlowData))
    for data in FlowData:
        if data['Hop_latency'] != []:
            data['Hop_latency'] = sum(data['Hop_latency'])/len(data['Hop_latency'])
        else:
            data['Hop_latency'] = None
        if data['Egress_time'] != []:
            data['Egress_time'] = sum(data['Egress_time'])/len(data['Egress_time'])
        else:
            data['Egress_time'] = None
        if data['Flow_latency'] != []:
            data['Flow_latency'] = sum(data['Flow_latency'])/len(data['Flow_latency'])
        else:
            data['Flow_latency'] = None
        if data['Sink_time'] != []:
            data['Sink_time'] = sum(data['Sink_time'])/len(data['Sink_time'])
        else:
            data['Sink_time'] = None
        if data['Port_tx_utilize'] != []:
            data['Port_tx_utilize'] = sum(data['Port_tx_utilize'])/len(data['Port_tx_utilize'])
        else:
            data['Port_tx_utilize'] = None
        if data['Queue_occupancy'] != []:
            data['Queue_occupancy'] = sum(data['Queue_occupancy'])/len(data['Queue_occupancy'])
        else:
            data['Queue_occupancy'] = None
    #print(FlowData[0])
    if not args.save_as_list:
        if args.file_name+'.pkl' in listdir('.'): #check if pkl file already exist.
            response=None
            while response not in ['y','n']:
                response=raw_input("flow data file is already existing. Do you wanna combine? (y/n) ")
            if response=='y':
                with open(args.file_name+'.pkl', 'r') as f:
                    OriFlowData=pkl.load(f)
                OriFlowData+=FlowData
                with open(args.file_name+'.pkl','w') as f:
                    pkl.dump(OriFlowData,f)
                print('saving on existing file Done!')
        else:
            with open(args.file_name+'.pkl','w') as f:
                pkl.dump(FlowData,f)
            print('saving on new file Done!')
    else:
        ListFlowData=[]
        for data in FlowData:
            list_data=[]
            list_data.append(data['Anomaly'])
            list_data.append(data['Path'])
            list_data.append(data['Protocol'])
            list_data.append(data['Duration'].total_seconds())
            list_data.append(data['Hop_latency'])
            list_data.append(data['Egress_time'])
            list_data.append(data['Flow_latency'])
            list_data.append(data['Sink_time'])
            list_data.append(data['Port_tx_utilize'])
            list_data.append(data['Queue_occupancy'])
            # 0 Anomaly : 0 or 1, int
            # 1 Path : List of string. Path could be []. ex)['11:22:12', '11:21:12']
            # 2 Protocol : int. 17 for UDP.
            # 3 Duration : datetime.timedelta object. max value could be given by datetime.timedelta(0,args.max_duration, 0)
            # 4 Hop_latency : None if not observed, else int.
            # 5 Egress_time : None if not observed, else int.
            # 6 Flow_latency : None if not observed, else int.
            # 7 Sink_time : None if not observed, else int.
            # 8 Port_tx_utilize : None if not observed, else int.
            # 9 Queue_occupancy : None if not observed, else int.
            assert(len(list_data)==10)
            ListFlowData.append(list_data)
        FlowData=ListFlowData
        if args.file_name+'_list.pkl' in listdir('.'): #check if pkl file already exist.
            response=None
            while response not in ['y','n']:
                response=raw_input("flow data file is already existing. Do you wanna combine? (y/n) ")
            if response=='y':
                with open(args.file_name+'_list.pkl', 'r') as f:
                    OriFlowData=pkl.load(f)
                OriFlowData+=FlowData
                with open(args.file_name+'_list.pkl','w') as f:
                    pkl.dump(OriFlowData,f)
                print('saving on existing file Done!')
        else:
            with open(args.file_name+'_list.pkl','w') as f:
                pkl.dump(FlowData,f)
            print('saving on new file Done!')

def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='example code to play with InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='localhost',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    parser.add_argument('--anomaly', action='store_true')
    parser.add_argument('--print_query', action='store_true')
    parser.add_argument('--save_as_list',action='store_true')
    parser.add_argument('--max_duration',type=float,default=2.0,
        help='max duration time of flow with sencond')
    parser.add_argument('--file_name', type=str, default='flow_data',
        help='file name to save the result.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args, host=args.host, port=args.port)