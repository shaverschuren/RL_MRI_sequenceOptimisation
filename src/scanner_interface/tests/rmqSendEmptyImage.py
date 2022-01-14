#!/usr/bin/env python3
# Send empty image to rmq
"""
Created on Fri Jan 11 10:00:33 2022

@author: tbruijnen
"""

import os
import sys
sys.path.append('/nfs/arch11/researchData/USER/tbruijne/Projects_Main/ReconSocket/recon-socket-repo/recon-socket/python_scripts/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pika
import numpy as np
import argparse
import dataobject_pb2 as pb
import time

def get_args():
    parser = argparse.ArgumentParser(description='RMQ unit test empty image transmission.')
    parser.add_argument('-m',metavar='machine_id',nargs='?',default="trumer",help='rtrabbit or trumer')
    args = parser.parse_args()
    return args

def main(machine_id):
    # Create an empty data-object
    X = 128
    Y = 128
    dataobject = pb_empty_object(X,Y)

    # Serialize to protobuf format
    result = dataobject.SerializeToString()
    #time.sleep(2)

    # Send away
    cred = pika.PlainCredentials('mquser','mquser')
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=machine_id,credentials=cred))
    channel = connection.channel()
    channel.basic_publish('reconsocket_image_dev','reconsocket.image',result,pika.BasicProperties(content_type='protobuf',delivery_mode=2))            
    connection.close()
    return 0

def pb_empty_object(X,Y):
    # Create a all-zeros image
    data_arr = np.zeros((X,Y),dtype=int)
    dim = data_arr.shape

    dataobject = pb.DataObject()

    ''' Add space object '''
    space = pb.SpaceObject()

    # mix
    mix_current = pb.rangeI32()
    mix_total = pb.rangeI32()
    mix_current.lower = 0
    mix_current.upper = 0
    mix_total.lower = 0
    mix_total.upper = 0
    space.mix.extend([mix_current, mix_total])

    # echo
    echo_current = pb.rangeI32()
    echo_total = pb.rangeI32()
    echo_current.lower = 0
    echo_current.upper = 0
    echo_total.lower = 0
    echo_total.upper = 0
    space.echo.extend([echo_current, echo_total])

    # loc
    loc_current = pb.rangeI32()
    loc_total = pb.rangeI32()
    loc_current.lower = 0
    loc_current.upper = 0
    loc_total.lower = 0
    loc_total.upper = 0
    space.location.extend([loc_current, loc_total])

    # row
    row_current = pb.rangeI32()
    row_total = pb.rangeI32()
    row_current.lower = 0
    row_current.upper = 0
    row_total.lower = 0
    row_total.upper = 0
    space.row.extend([row_current, row_total])

    # dyn
    dyn_current = pb.rangeI32()
    dyn_total = pb.rangeI32()
    dyn_current.lower = 0
    dyn_current.upper = 0
    dyn_total.lower = 0
    dyn_total.upper = 0
    space.dynamic.extend([dyn_current, dyn_total])

    # meas
    meas_current = pb.rangeI32()
    meas_total = pb.rangeI32()
    meas_current.lower = 0
    meas_current.upper = 0
    meas_total.lower = 0
    meas_total.upper = 0
    space.measurement.extend([meas_current, meas_total])

    # phase
    pha_current = pb.rangeI32()
    pha_total = pb.rangeI32()
    pha_current.lower = 0
    pha_current.upper = 0
    pha_total.lower = 0
    pha_total.upper = 0
    space.phase.extend([pha_current, pha_total])

    # extra
    ext_current = pb.rangeI32()
    ext_total = pb.rangeI32()
    ext_current.lower = 0
    ext_current.upper = 0
    ext_total.lower = 0
    ext_total.upper = 0
    space.extra_attr_value.extend([ext_current, ext_total])

    # X
    x_current = pb.rangeI32()
    x_total = pb.rangeI32()
    x_current.lower = int(-X/2)
    x_current.upper = int(X/2-1)
    x_total.lower = x_current.lower
    x_total.upper = x_current.upper
    space.x.extend([x_current, x_total])

    # Y
    y_current = pb.rangeI32()
    y_total = pb.rangeI32()
    y_current.lower = int(-Y/2)
    y_current.upper = int(Y/2-1)
    y_total.lower = y_current.lower
    y_total.upper = y_current.upper
    space.y.extend([y_current, y_total])

    # Z
    z_current = pb.rangeI32()
    z_total = pb.rangeI32()
    z_current.lower = 0
    z_current.upper = 0
    z_total.lower = 0
    z_total.upper = 0
    space.z.extend([z_current, z_total])

    # Channel
    cha_current = pb.rangeI32()
    cha_total = pb.rangeI32()
    cha_current.lower = 0
    cha_current.upper = 0
    cha_total.lower = -256
    cha_total.upper = 255
    space.channel.extend([cha_current, cha_total])

    # grad
    grad_current = pb.rangeI32()
    grad_total = pb.rangeI32()
    grad_current.lower = 0
    grad_current.upper = 0
    grad_total.lower = -256
    grad_total.upper = 255
    space.grad.extend([grad_current, grad_total])

    # rf
    rf_current = pb.rangeI32()
    rf_total = pb.rangeI32()
    rf_current.lower = 0
    rf_current.upper = 0
    rf_total.lower = -256
    rf_total.upper = 255
    space.rf.extend([rf_current, rf_total])

    # blade
    bla_current = pb.rangeI32()
    bla_total = pb.rangeI32()
    bla_current.lower = 0
    bla_current.upper = 0
    bla_total.lower = -256
    bla_total.upper = 255
    space.blade.extend([bla_current, bla_total])

    ''' Add empty image object '''
    # data
    data = pb.complexFloat32()
    data.real = 0.0
    data.imag = 0.0
    dataobject.data.append(data)

    # image - Rest of dim is assigned later on
    data_arr2 = np.ravel(data_arr)
    for n in range(len(data_arr2)):
        dataobject.image.append(data_arr2[n])

    # size_dim - batch size of data
    dataobject.size_dim.append(dim[0])
    dataobject.size_dim.append(dim[1])

    # size
    dataobject.size = 1

    # order
    dataobject.order = 3

    # space
    dataobject.space.append(space)

    # slice thickness
    dataobject.slice_thickness = 2.0

    # spacing
    dataobject.spacing_between_slices = 2.0

    # voxel size
    dataobject.voxel_sizes.append(2)
    dataobject.voxel_sizes.append(2)
    dataobject.voxel_sizes.append(2)

    # direc cosines
    dataobject.loc_ap_rl_fh_row_image_oris.append(1)
    dataobject.loc_ap_rl_fh_row_image_oris.append(0)
    dataobject.loc_ap_rl_fh_row_image_oris.append(0)
    dataobject.loc_ap_rl_fh_col_image_oris.append(0)
    dataobject.loc_ap_rl_fh_col_image_oris.append(1)
    dataobject.loc_ap_rl_fh_col_image_oris.append(0)

    # offcenters
    dataobject.loc_ap_rl_fh_offcentres.append(0)
    dataobject.loc_ap_rl_fh_offcentres.append(0)
    dataobject.loc_ap_rl_fh_offcentres.append(0)

    return dataobject

if __name__ == "__main__":
    args = get_args()
    main(args.m)

   
