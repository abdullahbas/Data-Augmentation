# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:09:43 2021

@author: trabz
"""
import random
import string

from timeit import default_timer as timer
import glob

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class Parser():

    def parse_etree_lxml(file):
        from lxml import etree as etree_lxml
        #parse xml
        xml_as_bytes = Parser.sample_xml('rb',file=file)
    
        timer_start = timer()
    
        print('[etree lxml] Starting to parse XML')
        
        tree = etree_lxml.fromstring(xml_as_bytes)
        ## Find <object> <object\> in the xml
        xml_etree_lxml = tree.findall('object')
        
        seconds = timer() - timer_start
    
        print(f'[etree lxml] Finished parsing XML in {seconds} seconds')
        return xml_etree_lxml
    def sample_xml(opts,file):
        """Return the sample XML file as a string."""
        with open(file, opts) as xml:
            return xml.read()
        
        
    def recursive_dict(element):
         return element.tag, \
                dict(map(Parser.recursive_dict, element)) or element.text
         #return all elements in xml with recursive way       
        
    def finalRun(path):    
        listw=glob.glob(path)
        #taking all path
        alls=[]
        finals=[]
        for xmls in listw:
            a0=Parser.parse_etree_lxml(xmls) # parse xmls
            for obj in a0:
                alls.append(Parser.recursive_dict(obj)[1]) #append all the dicts
                
            finals.append(alls)
            alls=[]
    def getValue(element,name):
        result=element.find(name).text
        return element.tag,name, result
    def myType(path,idx,classes=['bird','zebra']):
        objects=Parser.parse_etree_lxml(path)       
        boxes=[]
        labels=[]
        bbs=[]
        for obj in objects:
            xmin=int(Parser.getValue(obj.find('bndbox'),'xmin')[2])
            ymin=int(Parser.getValue(obj.find('bndbox'),'ymin')[2])
            xmax=int(Parser.getValue(obj.find('bndbox'),'xmax')[2])
            ymax=int(Parser.getValue(obj.find('bndbox'),'ymax')[2])
            boxes.append([xmin,ymin,xmax,ymax])
            bbs.append(BoundingBox(x1=xmin,y1=ymin,x2=xmax,y2=ymax))
            label=Parser.getValue(obj,'name')
            label= classes.index(label[2])
            labels.append(label)
            
        return {'image_id':idx,'label':labels,'bbox':boxes,'bbs':bbs}
    
    
            





