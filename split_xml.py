#-*- coding:utf-8 -*-
from xml.etree import cElementTree as et
import copy
import random

XML_PATH = './weibo.xml'
XML_TRAIN_PATH = './weibo_train.xml'
XML_TEST_PATH = './weibo_test.xml'

def num_getter():
  cnt = [-1]
  def _number_getter():
    cnt[0] += 1
    return cnt[0]
  return _number_getter

NUM_GETTER = num_getter()
TEST_PERCENTAGE = 0.1

def split_xml(xml_tree):
  root = xml_tree.getroot()
  weibos = map(lambda each:(NUM_GETTER(), each) , root.findall('weibo'))
  chosen_id = set(random.sample(range(len(weibos)), int(len(weibos) * TEST_PERCENTAGE)))
  chosen_weibo = filter(lambda each: each[0] in chosen_id, weibos)
  train_tree = copy.deepcopy(xml_tree)

  map(lambda each: train_tree.getroot().remove(each), train_tree.findall('weibo'))
  map(lambda each: train_tree.getroot().append(each[1]), chosen_weibo)
  map(lambda each: xml_tree.getroot().remove(each[1]), chosen_weibo)

  train_tree.write(XML_TEST_PATH, encoding='utf-8')
  xml_tree.write(XML_TRAIN_PATH, encoding='utf-8')

# 切分数据集(10%作为test,　90%作为train)
if __name__ == '__main__':
  split_xml(et.parse(XML_PATH))