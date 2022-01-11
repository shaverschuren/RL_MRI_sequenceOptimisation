#!/usr/bin/python3
''' Some remarks, protobuf (protoc) needs to be installed on your system '''

import pika
import os
import sys
sys.path.append(str(os.path.dirname(os.path.realpath(__file__))))
import argparse
import dataobject_pb2 as pb # pb = protobuf 
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='RMQ receive image inline python interface.')
    parser.add_argument('-m',metavar='machine_id',nargs='?',default="rtrabbit",help='rtrabbit or trumer')
    parser.add_argument('-ut',metavar='unit_test',type=bool,nargs='?',default=False,const=True,help='check whether server connection can be established, test is succesful if "Image received" is printed.')
    args = parser.parse_args()
    return args

def callback_image(ch, method, properties, body):
    global img_number
    img_number += 1
    print("Image number",img_number,"received.")
    dataobject = pb.DataObject()
    dataobject.ParseFromString(body)
    image_data = dataobject.image
    image_data = np.reshape(image_data,dataobject.size_dim)

def rmq_setup_channel(machine_id):
    cred = pika.PlainCredentials('mquser','mquser')
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=machine_id,credentials=cred))    
    channel = connection.channel()
    channel.queue_declare(queue='reconsocket_image_dev')
    channel.queue_bind(exchange='reconsocket_image_dev', queue='reconsocket_image_dev',routing_key='reconsocket.image')
    channel.basic_consume(queue='reconsocket_image_dev', on_message_callback=callback_image, auto_ack=True)
    return channel

def rmq_unit_test(machine_id):
    script_path = os.path.dirname(os.path.realpath(__file__))
    test_str = (sys.executable,' ',str(script_path),'/rmqSendEmptyImage.py',' -m ',machine_id,' &')
    os.system(''.join(test_str))  
    return

if __name__ == "__main__":
    img_number = 0
    args = get_args()
    channel = rmq_setup_channel(args.m)  
    if args.ut:
        rmq_unit_test(args.m)
    print('[*] Waiting for images on',args.m,'. Type CTRL+C to exit')
    channel.start_consuming()

