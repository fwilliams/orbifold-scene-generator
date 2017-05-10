import argparse
import numpy as np
from lxml import etree
import time
from time import localtime, strftime
import sys
import os
import copy


def gen_scene_xml(xml_filename, xml_frametime):
    scene_doc = etree.parse(xml_filename)
    root = scene_doc.getroot()

    sensor = root.find("sensor")

    frametime = ""
    frametime += str(xml_frametime)

    component_format = sensor.xpath("//float[@name='shutterClose']")

    for c in component_format:
        c.set("value", frametime)

    component_format = sensor.xpath("//float[@name='shutterOpen']")
    for c in component_format:
        c.set("value", frametime)

    return root

argparser = argparse.ArgumentParser()
argparser.add_argument("filename", help="The name of the scene file to start.", type=str)
argparser.add_argument("shutterOpen", help="int, the time of animation start", type=int)
argparser.add_argument("shutterClose", help="int, the time of animation end", type=int)

args = argparser.parse_args()

timeRange = args.shutterClose - args.shutterOpen + 1

timestamp = strftime("%d%b%Y_%H_%M_%S", localtime())

output_dir = "./animation_%s_%s" % (os.path.basename(args.filename), timestamp)
output_dir = os.path.realpath(output_dir)
os.mkdir(output_dir)

print("Generating animation scenes...")

for step in range(timeRange):
    frameStamp = args.shutterOpen + step
    scene_doc = gen_scene_xml(args.filename, frameStamp)

    frameStamp = args.shutterOpen+step
    with open(os.path.join(output_dir, "animation_%d.xml" %frameStamp ), "w+") as f:
        f.write(etree.tostring(scene_doc, pretty_print=True))

print("Saved scene data to %s" % output_dir)