from __future__ import division
import time
import numpy as np
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui
import sys
import argparse
import warnings
import json
import paho.mqtt.client as mqtt
import datetime

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("-f", default='/dev/ttyACM1', help="serial port", type=str)
parser.add_argument("-MQTT", default="192.168.1.3", help="mqtt broker ex 192.168.1.3", type=str)
parser.add_argument("-topic_robot", default="robot_control", help="robot control topic", type=str)
parser.add_argument("-topic_terabee", default="terabee", help="terabee topic", type=str)
parser.add_argument("-topic_picamera", default="picamera", help="picamera topic", type=str)
parser.add_argument("-topic_fused", default="fused", help="fused data topic", type=str)
parser.add_argument("-RID", default=0, help="ROBOT IDENTIFIER", type=int)
parser.add_argument("-samp", default=2, help="observations per action", type=int)
args = parser.parse_args()


# Change the configuration file name
configFileName = args.f
readySerial = False
CLIport = {}
transStart = np.uint8(128)
transEnd = np.uint8(129)
requestEcho = np.uint8(0)
echo = np.uint8(1)
requestSupplyVoltage = np.uint8(10);
supplyVoltage = np.uint8(11);
requestCrawlForward = np.uint8(80)
requestCrawlBackward = np.uint8(82)
requestCrawlLeft = np.uint8(84)
requestCrawlRight = np.uint8(86)
requestTurnLeft = np.uint8(88)
requestTurnRight = np.uint8(90)
requestActiveMode = np.uint8(92)
requestSleepMode = np.uint8(94)
requestSwitchMode = np.uint8(96)
orderStart = np.uint8(21)
orderDone = np.uint8(23)

requestCrawl = np.uint8(110) # [order] [64 + x] [64 + y] [64 + angle]
requestChangeBodyHeight = np.uint8(112)  # [order] [64 + height]
requestMoveBody = np.uint8(114)          # [order] [64 + x] [64 + y] [64 + z]
requestRotateBody = np.uint8(116)        # [order] [64 + x] [64 + y] [64 + z]
requestTwistBody = np.uint8(118)         # [order] [64 + xMove] [64 + yMove] [64 + zMove] [64 + xRotate] [64 + yRotate] [64 + zRotate]

command_active = bytearray([transStart, requestActiveMode, transEnd])
command_forward = bytearray([transStart, requestCrawlForward, transEnd])
command_backward = bytearray([transStart, requestCrawlBackward, transEnd])
command_left = bytearray([transStart, requestCrawlLeft, transEnd])
command_right = bytearray([transStart, requestCrawlRight, transEnd])
command_sleep = bytearray([transStart, requestSleepMode, transEnd])
command_supply = bytearray([transStart, requestSupplyVoltage, transEnd])
command_turnleft = bytearray([transStart, requestTurnLeft, transEnd])
command_turnright = bytearray([transStart, requestTurnRight, transEnd])

byteBuffer = np.zeros(3,dtype = 'uint8')
command_trace = {}

def on_release(key):
    global mqttc
    global CLIport
    global command_sleep
    if key == keyboard.Key.esc:
        mqttc.loop_stop()
        mqttc.disconnect()
        blockingCommand(command_sleep, CLIport)
        CLIport.close()
        sys.exit()

def voltageCommand(command, CLIport):
    CLIport.reset_input_buffer()
    CLIport.write(command)
    readBuffer = CLIport.read(5)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    if byteVec[1] == supplyVoltage:
        status = True
        voltage_measured = np.float_((byteVec[2] * 128 + byteVec[3]) / 100)
        print("Supply Voltage:", voltage_measured)
    else:
        status = False
    return status

def blockingCommand(command, CLIport):
    CLIport.reset_input_buffer()
    CLIport.write(command)
    readBuffer = CLIport.read(6)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    if byteVec[4] == orderDone:
        status = True
    else:
        status = False
    return status

def blockingCommand_complex(command, CLIport, param):
    global transEnd
    global transStart
    param2 = np.uint8(param)
    CLIport.reset_input_buffer()
    command2 = bytearray([transStart, command, param2, transEnd])
    CLIport.write(command2)
    readBuffer = CLIport.read(6)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    if byteVec[4] == orderDone:
        status = True
    else:
        status = False
    return status

# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    global CLIport
    global command_active

    CLIport = serial.Serial(configFileName, 115200)
    CLIport.reset_input_buffer()
    time.sleep(3)
    command_echo = bytearray([transStart, requestEcho, transEnd])
    CLIport.write(command_echo)
    time.sleep(2)
    readBuffer = CLIport.read(CLIport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    if byteVec[1] == echo:
        print('connected')
        blockingCommand(command_active, CLIport)
        ready_serial = True
    else:
        ready_serial = False
    return CLIport, ready_serial

def on_message(mqtt_client, obj, msg):
    print("on_message()")

# ------------------------------------------------------------------
def robot_control_callback(client, userdata, message):
    # print("ok")
    command_mqtt = int(message.payload)
    global readySerial
    global CLIport
    global command_active
    global command_backward
    global command_left
    global command_right
    global command_forward
    global command_sleep
    global command_supply
    global mqttc
    global command_trace

    # evo_64px.run()
    # detObj_terabee = {}
    # detObj_picamera = {}
    if command_mqtt == 11:
        detObj_fused = {}
        time.sleep(0.5)  # sleep 0.5 sec to limit sensing impaiments caused by robot vibrations
        with picamera.PiCamera() as camera:
            camera.resolution = (320, 240)
            camera.framerate = 24
            # output_g = np.empty((240, 320, 3, samples), dtype=np.uint8)
            # output = np.empty((240, 320, 3, args.samp), dtype=np.uint8)
            output_partial = np.empty((240, 320, 3), dtype=np.uint8)
            # depth_array = np.empty((8, 8, args.samp))
            # terabee = np.empty((8, 8, samples), dtype=np.uint16)

            time.sleep(0.1)
            camera.capture(output_partial, 'rgb', use_video_port=True)
            # detObj_picamera['time'] = datetime.datetime.now().isoformat()
            detObj_fused['camera'] = np.asarray(output_partial).tolist()
            # detObj_picamera['id'] = rob_id
            detObj_fused['command'] = command_mqtt
            # try:
            #    mqttc.publish(args.topic_picamera, json.dumps(detObj_picamera))
            # except:
            #    print("error sending")
            #    mqttc.disconnect()

            depth_array = evo_64px.run(args.samp)
            detObj_fused['time'] = datetime.datetime.now().isoformat()
            detObj_fused['terabee'] = depth_array.tolist()
            detObj_fused['id'] = rob_id
            # detObj_terabee['command'] = command_mqtt

            try:
                mqttc.publish(args.topic_fused, json.dumps(detObj_fused))
            except:
                print("error sending")
                mqttc.disconnect()

    if readySerial:
        if command_mqtt == 1:
            CLIport.reset_input_buffer()
            blockingCommand(command_forward, CLIport)
            blockingCommand(command_forward, CLIport)
            blockingCommand(command_forward, CLIport)
            blockingCommand(command_forward, CLIport)
            blockingCommand(command_forward, CLIport)
            blockingCommand(command_forward, CLIport)
            blockingCommand(command_forward, CLIport)
            blockingCommand(command_forward, CLIport)
            blockingCommand(command_forward, CLIport)
        elif command_mqtt == 2:
            CLIport.reset_input_buffer()
            blockingCommand(command_backward, CLIport)
            blockingCommand(command_backward, CLIport)
            blockingCommand(command_backward, CLIport)
            blockingCommand(command_backward, CLIport)
            blockingCommand(command_backward, CLIport)
            blockingCommand(command_backward, CLIport)
            blockingCommand(command_backward, CLIport)
            blockingCommand(command_backward, CLIport)
            blockingCommand(command_backward, CLIport)
        elif command_mqtt == 3:
            CLIport.reset_input_buffer()
            blockingCommand(command_left, CLIport)
            blockingCommand(command_left, CLIport)
            blockingCommand(command_left, CLIport)
            blockingCommand(command_left, CLIport)
            blockingCommand(command_left, CLIport)
            blockingCommand(command_left, CLIport)
            blockingCommand(command_left, CLIport)
            blockingCommand(command_left, CLIport)
            blockingCommand(command_left, CLIport)
        elif command_mqtt == 4:
            CLIport.reset_input_buffer()
            blockingCommand(command_right, CLIport)
            blockingCommand(command_right, CLIport)
            blockingCommand(command_right, CLIport)
            blockingCommand(command_right, CLIport)
            blockingCommand(command_right, CLIport)
            blockingCommand(command_right, CLIport)
            blockingCommand(command_right, CLIport)
            blockingCommand(command_right, CLIport)
            blockingCommand(command_right, CLIport)
        elif command_mqtt == 5:
            CLIport.reset_input_buffer()
            blockingCommand(command_turnleft, CLIport)
        elif command_mqtt == 6:
            CLIport.reset_input_buffer()
            blockingCommand(command_turnright, CLIport)
        elif command_mqtt == 7:
            CLIport.reset_input_buffer()
            blockingCommand(command_forward, CLIport)
        elif command_mqtt == 8:
            CLIport.reset_input_buffer()
            blockingCommand(command_backward, CLIport)
        elif command_mqtt == 9:
            CLIport.reset_input_buffer()
            blockingCommand(command_left, CLIport)
        elif command_mqtt == 10:
            CLIport.reset_input_buffer()
            blockingCommand(command_right, CLIport)
        elif command_mqtt == 0:
            # get Voltage
            CLIport.reset_input_buffer()
            voltageCommand(command_supply, CLIport)
        elif command_mqtt == 11:
            #do nothing
            CLIport.reset_input_buffer()
        else:
            print("command not recognized")
    else:
        CLIport.close()


# -------------------------    MAIN   -----------------------------------------
if __name__ == "__main__":
    
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "ttyACM" in p[1] and ":5740" not in p[1]:
            configFileName = p[0]
            print(configFileName)
    if configFileName is None:
        print("Robot not found. Please Check connections.")
        exit()

    # Configurate the serial port
    evo_64px = Evo_64px()
    CLIport, readySerial = serialConfig(configFileName)
    if readySerial:
        MQTT_broker = args.MQTT
        rob_id = args.RID
        client_py = "robot_hexapod " + str(rob_id)
        mqttc = mqtt.Client(client_id=client_py, clean_session=True)
        mqttc.connect(host=MQTT_broker, port=1885, keepalive=60)
        topic_control_robot = args.topic_robot + str(rob_id)
        mqttc.subscribe(topic_control_robot, qos=0)
        mqttc.message_callback_add(topic_control_robot, robot_control_callback)
        mqttc.loop_start()

    listener = keyboard.Listener(
        on_release=on_release)
    listener.start()





