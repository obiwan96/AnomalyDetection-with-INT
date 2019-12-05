#!/usr/bin/python

from bcc import BPF
from influxdb import InfluxDBClient
from ipaddress import IPv4Address
import threading
import ctypes as ct

class InDBCollector(object):
    """docstring for InDBCollector"""

    def __init__(self, max_int_hop=6, debug_mode=0, int_dst_port=54321, host="localhost",
        database="INTdatabase", event_mode="THRESHOLD"):
        super(InDBCollector, self).__init__()

        self.MAX_INT_HOP = max_int_hop
        self.SERVER_MODE = "INFLUXDB"
        self.INT_DST_PORT = int_dst_port
        self.EVENT_MODE = event_mode

        self.ifaces = set()

        #load eBPF program
        self.bpf_collector = BPF(src_file="BPFCollector.c", debug=0,
            cflags=["-w", 
                    "-D_MAX_INT_HOP=%s" % self.MAX_INT_HOP,
                    "-D_INT_DST_PORT=%s" % self.INT_DST_PORT,
                    "-D_EVENT_MODE=%s" % self.EVENT_MODE,
                    "-D_SERVER_MODE=%s" % self.SERVER_MODE])
        self.fn_collector = self.bpf_collector.load_func("collector", BPF.XDP)

        # get all the info table
        self.tb_flow  = self.bpf_collector.get_table("tb_flow")
        self.tb_egr   = self.bpf_collector.get_table("tb_egr")
        self.tb_queue = self.bpf_collector.get_table("tb_queue")

        self.flow_paths = {}

        self.lock = threading.Lock()
        self.event_data = []

        self.client = InfluxDBClient(host=host, database=database)

        self.debug_mode = debug_mode


    def attach_iface(self, iface):
        if iface in self.ifaces:
            print "already attached to ", iface
            return
        self.bpf_collector.attach_xdp(iface, self.fn_collector, 0)
        self.ifaces.add(iface)

    def detach_iface(self, iface):
        if iface not in self.ifaces:
            print "no program attached to ", iface
            return
        self.bpf_collector.remove_xdp(iface, 0)
        self.ifaces.remove(iface)

    def detach_all_iface(self):
        for iface in self.ifaces:
            self.bpf_collector.remove_xdp(iface, 0)
        self.ifaces = set()

        
    def open_events(self):
        def _process_event(ctx, data, size):
            class Event(ct.Structure):
                _fields_ =  [("src_ip", ct.c_uint32),
                             ("dst_ip", ct.c_uint32),
                             ("src_port", ct.c_ushort),
                             ("dst_port", ct.c_ushort),
                             ("ip_proto", ct.c_ushort),
                             
                             # ("pkt_cnt", ct.c_uint64),
                             # ("byte_cnt", ct.c_uint64),

                             ("num_INT_hop", ct.c_ubyte),

                             ("sw_ids", ct.c_uint32 * self.MAX_INT_HOP),
                             ("in_port_ids", ct.c_uint16 * self.MAX_INT_HOP),
                             ("e_port_ids", ct.c_uint16 * self.MAX_INT_HOP),
                             ("hop_latencies", ct.c_uint32 * self.MAX_INT_HOP),
                             ("queue_ids", ct.c_uint16 * self.MAX_INT_HOP),
                             ("queue_occups", ct.c_uint16 * self.MAX_INT_HOP),
                             # ("ingr_times", ct.c_uint32 * self.MAX_INT_HOP),
                             ("egr_times", ct.c_uint32 * self.MAX_INT_HOP),
                             ("lv2_in_e_port_ids", ct.c_uint32 * self.MAX_INT_HOP),
                             ("tx_utilizes", ct.c_uint32 * self.MAX_INT_HOP),

                             ("flow_latency", ct.c_uint32),
                             ("flow_sink_time", ct.c_uint32),

                             ("is_n_flow", ct.c_ubyte),
                             # ("is_n_hop_latency", ct.c_ubyte),
                             # ("is_n_queue_occup", ct.c_ubyte),
                             # ("is_n_queue_congest", ct.c_ubyte),
                             # ("is_n_tx_utilize", ct.c_ubyte),

                             ("is_flow", ct.c_ubyte),
                             ("is_hop_latency", ct.c_ubyte),
                             ("is_queue_occup", ct.c_ubyte),
                             # ("is_queue_congest", ct.c_ubyte),
                             ("is_tx_utilize", ct.c_ubyte)
                             ]

            event = ct.cast(data, ct.POINTER(Event)).contents
            # push data
            
            event_data = []
            
            if event.is_n_flow or event.is_flow:
                path_str = ":".join(str(event.sw_ids[i]) for i in reversed(range(0, event.num_INT_hop)))
                event_single_data={"measurement": "flow {0}->{1}".format(
                                                    str(IPv4Address(event.src_ip)),
                                                    str(IPv4Address(event.dst_ip)),
                                                    ),
                                    "time": event.flow_sink_time,
                                    "fields": {
                                        "pkt_cnt"  : event.pkt_cnt,
                                        "byte_cnt" : event.byte_cnt,
                                        "flow_latency" : event.flow_latency,
                                        "path": path_str,
                                        "ip_proto" : event.ip_proto
                                    }}
                if event.is_hop_latency:
                    sum_hop_latencies=0
                    num_hop_latencies=0
                    for i in range(0, event.num_INT_hop):
                        if ((event.is_hop_latency >> i) & 0x01):
                            sum_hop_latencies+=event.hop_latencies[i]
                            num_hop_latencies+=1
                    event_single_data["fields"]["hop_latency"]=sum_hop_latencies/num_hop_latencies


                if event.is_tx_utilize:
                    sum_tx_utilize=0
                    num_tx_utilize=0
                    for i in range(0, event.num_INT_hop):
                        if ((event.is_tx_utilize >> i) & 0x01):
                            sum_tx_utilize+=event.tx_utilizes[i]
                            num_tx_utilize+=1
                    event_single_data["fields"]["port_tx_utilize"]=sum_tx_utilize/num_tx_utilize

                if event.is_queue_occup:
                    sum_queue_occup=0
                    num_queue_occup=0
                    for i in range(0, event.num_INT_hop):
                        if ((event.is_queue_occup >> i) & 0x01):
                            sum_queue_occup+=event.queue_occups[i]
                            num_queue_occup+=1
                    event_single_data["fields"]["queue_occupancy"]=sum_queue_occup/num_queue_occup
                event_data.append(event_single_data)

            # if event.is_queue_congest:
            #     for i in range(0, event.num_INT_hop):
            #         if ((event.is_queue_congest >> i) & 0x01):
            #             event_data.append({"measurement": "queue_congestion,sw_id={0},queue_id={1}".format(
            #                                                 event.sw_ids[i], event.queue_ids[i]),
            #                                 "time": event.egr_times[i],
            #                                 "fields": {
            #                                     "value": event.queue_congests[i]
            #                                 }
            #                             })

            # self.client.write_points(points=event_data)
            self.lock.acquire()
            self.event_data.extend(event_data)
            self.lock.release()


            # Print event data for debug
            if self.debug_mode==1:
                print "*" * 20
                for field_name, field_type in event._fields_:
                    field_arr = getattr(event, field_name)
                    if field_name in ["sw_ids","in_port_ids","e_port_ids","hop_latencies",
                                    "queue_occups", "queue_ids","egr_times",
                                    "queue_congests","tx_utilizes"]:

                        _len = len(field_arr)
                        s = ""
                        for e in field_arr:
                            s = s+str(e)+", " 
                        print field_name+": ", s
                    else:
                        print field_name+": ", field_arr

        self.bpf_collector["events"].open_perf_buffer(_process_event, page_cnt=512)

    
    def poll_events(self):
        self.bpf_collector.kprobe_poll()

    def collect_data(self):
        # json_str = json.dumps(self.tb_egr.items())
        # print json_str

        data = []

        for (flow_id, flow_info) in self.tb_flow.items():
            path_str = ":".join(str(flow_info.sw_ids[i]) for i in reversed(range(0, flow_info.num_INT_hop)))
            sigle_data={"measurement": "flow {0}->{1}".format(
                                                    str(IPv4Address(flow_id.src_ip)),
                                                    str(IPv4Address(flow_id.dst_ip)) 
                                                    ),
                            "time": flow_info.flow_sink_time,
                            "fields": {
                                "pkt_cnt"  : flow_info.pkt_cnt,
                                "byte_cnt" : flow_info.byte_cnt,
                                "flow_latency" : flow_info.flow_latency,
                                "ip_proto" : flow_id.ip_proto,
                                "path" : path_str
                            }
                        }

            if flow_info.is_hop_latency:
                single_dta["fields"]["hop_latency"]=sum(flow_info.hop_latencies)/flow_info.num_INT_hop
            data.append(single_dta)


        for (egr_id, egr_info) in self.tb_egr.items():
            for single_data in data:
                if egr_id.sw_id in single_data["fields"]["path"]:
                    if "port_tx_utilize" in sigle_data["fields"]:
                        single_data["fields"]["port_tx_utilize"].append(egr_info.tx_utilize)
                    else:
                        single_data["fields"]["port_tx_utilize"]=[egr_info.tx_utilize]

        for (queue_id, queue_info) in self.tb_queue.items():
            for single_data in data:
                if egr_id.sw_id in single_data["fields"]["path"]:
                    if "queue_occupancy" in single_data["fields"]:
                        single_data["fields"]["queue_occupancy"].append(egr_info.occup)
                    else:
                        single_data["fields"]["queue_occupancy"]=[egr_info.occup]

            # data.append({"measurement": "queue_congestion,sw_id={0},queue_id={1}".format(
            #                             queue_id.sw_id, queue_id.q_id),
            #                 "time": queue_info.q_time,
            #                 "fields": {
            #                     "value": queue_info.congest
            #                 }
            #             })


        return data




#---------------------------------------------------------------------------
# if __name__ == "__main__":
