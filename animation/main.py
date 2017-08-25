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

    ###setting the shutterOpen and shutterClose value
    sensor = root.find("sensor")

    frametime = ""
    frametime += str(xml_frametime)

    component_format = sensor.xpath("//float[@name='shutterClose']")

    for c in component_format:
        c.set("value", frametime)

    component_format = sensor.xpath("//float[@name='shutterOpen']")
    for c in component_format:
        c.set("value", frametime)

    ###move all the transformations inside animation
    shape_nodes = root.findall(".//shape")
    for node in shape_nodes:
        animation = node.find("animation")

        if animation is not None:
            transform = node.find("transform")
            matrix = transform.find("matrix")

            animTransforms = animation.findall("transform")
            for at in animTransforms:
                matrix1 = copy.deepcopy(matrix)
                at.append(matrix1)

            node.remove(transform)
            ###print(node)
        else:
            continue

    #print(shape_nodes)
    return root

argparser = argparse.ArgumentParser()
argparser.add_argument("filename", help="The name of the scene file to start.", type=str)
argparser.add_argument("shutterOpen", help="int, the time of animation start", type=int)
argparser.add_argument("shutterClose", help="int, the time of animation end", type=int)
argparser.add_argument("batchNumber", help="int, batch number of scenes into one folder, 0 means not batching", type=int)

args = argparser.parse_args()

timeRange = args.shutterClose - args.shutterOpen + 1

timestamp = strftime("%d%b%Y_%H_%M_%S", localtime())

output_dir = "./animation_%s_%s" % (os.path.basename(args.filename), timestamp)
output_dir = os.path.realpath(output_dir)
os.mkdir(output_dir)

print("Generating animation scenes...")

if args.batchNumber == 0:
    print("not batching outputs")
else:
    batchNumer = (timeRange+args.batchNumber-1) // args.batchNumber
    for b in range(batchNumer):
        output_dir = os.path.realpath(output_dir)
        output_barch_dir = output_dir + "/batch_%s" % b
        os.mkdir(output_barch_dir)


for step in range(timeRange):
    frameStamp = args.shutterOpen + step
    scene_doc = gen_scene_xml(args.filename, frameStamp)

    if args.batchNumber == 0:
        output_b_dir = output_dir
    else:
        batchNumber = step / args.batchNumber
        output_b_dir = output_dir + "/batch_%s" %batchNumber

    frameStamp = args.shutterOpen+step
    with open(os.path.join(output_b_dir, "animation_%03d.xml" %frameStamp ), "w+") as f:
        f.write(etree.tostring(scene_doc, pretty_print=True))

print("Saved scene data to %s" % output_dir)