#-*- coding:utf-8 -*-
from xml.etree import cElementTree as et
from collections import Counter
import jieba.posseg
import bayes

XML_PATH = './weibo.xml'
XML_TRAIN_PATH = './weibo_train.xml'
XML_TEST_PATH = './weibo_test.xml'
PICKLES_CATA = './training_pickles/'
WEIBO_EMOTION_DICT = {
  'none' : 0,
  'like' : 1,
  'sadness' : -1,
  'disgust' : -1,
  'anger' : -1,
  'surprise' : 1,
  'fear' : -1,
  'happiness' : 1
}

class ListWithLinkExtend(list):
  def extend(self, value):
    super(ListWithLinkExtend, self).extend(value)
    return self

# 读文件
def read_file(path):
  content = ''
  with open(path, 'r') as p:
    content = reduce(lambda x,y:x+y, p)
  return content

def adder():
  cnt = [-1]
  def _adder():
    cnt[0] += 1
    return cnt[0]
  return _adder

def _turn_to_same(each, a, b):
  if each == a:
    each = b
  return each

def get_content_emotion(sentence):
  if sentence.attrib['emotion_tag'] == 'N':
    return ['none']
  try:
    return [sentence.attrib['emotion-1-type'], sentence.attrib['emotion-2-type']]
  except KeyError:
    return [sentence.attrib['emotion-1-type']]

def get_dataset(path):
  xml_tree = et.parse(path)
  root = xml_tree.getroot()

  # weibo_emotion[i] : weibo[i]'s emotion
  weibo_emotion = map(lambda weibo: weibo.attrib['emotion-type'], root.findall('weibo'))
  # weibo_content[i][j] : weibo[i], j'th sentence
  weibo_content = map(lambda weibo: map(lambda sentence: sentence, weibo.findall('sentence')), root.findall('weibo'))

  # weibo_content_emotion[i][j] : weibo[i], j'th sentence's emotion
  weibo_content_emotion = map(lambda weibo: map(get_content_emotion, weibo), weibo_content)
  # print set(weibo_content_emotion)

  # weibo_content_set
  weibo_content_set = list(reduce(lambda x, y: x | y,
                      map(lambda weibo: reduce(lambda x, y: x | y, weibo),
                      map(lambda weibo: map(lambda sentence:set(sentence), weibo), weibo_content_emotion))))

  weibo_content = map(lambda weibo: map(lambda sentence: map(lambda each: each, sentence), weibo),
                  map(lambda weibo: map(lambda sentence:
                  [i for i in jieba.posseg.cut(sentence.text)], weibo), weibo_content))
  weibo_content = map(lambda weibo: map(lambda sentence:
                  filter(lambda word: word.flag != 'x', sentence), weibo), weibo_content)
  weibo_content = map(lambda weibo: map(lambda sentence:
                  map(lambda each:each.word, sentence), weibo), weibo_content)
  # weibo_content[i][j][k] : weibo[i], j'th sentence k'th word (splited word)
  weibo_content = map(lambda weibo: map(lambda sentence:
                                        reduce(lambda x, y: x + ' ' + y, sentence), weibo), weibo_content)
  weibo_content = map(lambda weibo: map(lambda sentence: sentence.split(' '), weibo), weibo_content)

  return weibo_emotion, weibo_content, weibo_content_emotion

def convert_to_mat_and_cat(bow, weibo_content, weibo_content_emotion):
  def _word_to_vec(data):
    vec = [0] * len(bow)
    for word in data:
      if word in bow:
        vec[bow.index(word)] += 1
    return vec
  train_mat = map(lambda weibo: map(_word_to_vec, weibo), weibo_content)
  train_cat = map(lambda weibo: map(lambda each: WEIBO_EMOTION_DICT[each[0]], weibo), weibo_content_emotion)
  train_mat = list(reduce(lambda a, b: ListWithLinkExtend(a).extend(ListWithLinkExtend(b)), train_mat))
  train_cat = list(reduce(lambda a, b: ListWithLinkExtend(a).extend(ListWithLinkExtend(b)), train_cat))
  return train_mat, train_cat

def process_training_xml(weibo_emotion, weibo_content, weibo_content_emotion):
  bow = []
  map(lambda weibo: map(lambda sentence: map(lambda each: bow.append(each), sentence), weibo), weibo_content)
  bow = list(set(bow))
  # word to vec
  train_mat, train_cat = convert_to_mat_and_cat(bow, weibo_content, weibo_content_emotion)

  # first_level_mat , first_level_cat : 有无感情色彩(1,-1有，0无)
  first_level_mat = train_mat
  first_level_cat = map(lambda each: _turn_to_same(each, -1, 1), train_cat)
  l1_p0_vect, l1_p1_vect, p_emotion = bayes.trainNB0(first_level_mat, first_level_cat)

  # second_level_mat , second_level_cat : 感情色彩积极消极(1积极-1消极，归为1积极，0消极)
  train_combile = zip(train_mat, train_cat)
  second_level_tmp = filter(lambda each: each[1]!=0, train_combile)
  second_level_mat = map(lambda each: each[0], second_level_tmp)
  second_level_cat = map(lambda each: each[1], second_level_tmp)
  second_level_cat = map(lambda each: _turn_to_same(each, -1, 0), second_level_cat)
  l2_p0_vect, l2_p1_vect, p_positive = bayes.trainNB0(second_level_mat, second_level_cat)

  return bow, l1_p0_vect, l1_p1_vect, p_emotion, l2_p0_vect, l2_p1_vect, p_positive


if __name__ == '__main__':
  weibo_emotion, weibo_content, weibo_content_emotion = get_dataset(XML_TRAIN_PATH)
  bow, l1_p0_vect, l1_p1_vect, p_emotion, l2_p0_vect, l2_p1_vect, p_positive = \
    process_training_xml(weibo_emotion, weibo_content, weibo_content_emotion)

  test_weibo_emotion, test_content, test_content_emotion = get_dataset(XML_TEST_PATH)
  test_mat, test_cat = convert_to_mat_and_cat(bow, test_content, test_content_emotion)

  # 进行两种情感的分类
  l1_pre_cat = map(lambda each: bayes.classifyNB(each, l1_p0_vect,l1_p1_vect, p_emotion), test_mat)
  l1_pre_cat_cnt = adder()
  test_content = map(lambda weibo: map(lambda sentence: (sentence, l1_pre_cat[l1_pre_cat_cnt()]), weibo), test_content)

  l2_pre_cat = map(lambda each: bayes.classifyNB(each, l2_p0_vect, l2_p1_vect, p_positive), test_mat)
  l2_pre_cat_cnt = adder()
  def _mark_emotion(each):
    if each[1] == 0:
      return each
    ret = l2_pre_cat[l2_pre_cat_cnt()]
    if ret == 1:
      return (each[0], 1)
    else:
      return (each[0], -1)

  test_content = map(lambda weibo: map(_mark_emotion, weibo), test_content)

  # 汇总两种分类结果
  sum_test_label = map(lambda weibo: map(lambda each: each[1], weibo), test_content)
  # 投票表决，如果感情全部相同则认为是中性
  sum_test_label = map(lambda weibo: Counter(weibo).most_common(1)[0], sum_test_label)
  test_weibo_emotion = map(lambda key: WEIBO_EMOTION_DICT[key], test_weibo_emotion)
  ret = 0
  for i in range(0, len(sum_test_label)):
    print 'Predict : ', sum_test_label[i][0], ' Actually : ', test_weibo_emotion[i]
    if (sum_test_label[i][0] > 0 and test_weibo_emotion[i] > 0) or (sum_test_label[i][0] < 0 and test_weibo_emotion[i] < 0) or (sum_test_label[i][0] == 0 and test_weibo_emotion[i] == 0):
      ret += 1
  print float(ret) / float(len(sum_test_label))

# 词典就是train test中的词语，每句话划分成一个大小为len(dic)的向量
# 训练算法：双层bayes，第一层区分一句话有、无感情色彩；第二层对有感情色彩的加以区分，区分是何种感情色彩。
# 最后使用每句话的分类，表决得出该微博信息的感情。
