import numpy as np
import jieba


# ==============判断char是否是乱码===================
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True       
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('，','。','：','？','“','”','！','；','、','《','》','——','=','*','@','￥','?','.'):
            return True
    return False

class GenData():
	def __init__(self):
		super(GenData, self).__init__()
		# 特殊字符
		self.SOURCE_CODES = ['<PAD>', '<UNK>']
		self.TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符
		self._init_data()
		self._init_vocab()
		self._init_num_data()


	def _init_data(self):
		# ========读取原始数据========
		with open('question', 'r', encoding='utf-8') as f:
			data = f.read()
			self.input_list = [list(jieba.cut(line)) for line in data.split('\n')]#即通过jieba将每一行分成矩阵
			#[['\ufeff', '今天', '吃', '了', '吗'], ['你', '在', '干什么'], ['晚上', '下雨', '吗'], ['你', '叫', '什么', '名字'],

		with open('answer', 'r', encoding='utf-8') as a:
			data = a.read()
			self.output_list = [list(jieba.cut(line)) for line in data.split('\n')]

		self.input_list = [[char for char in line] for line in self.input_list]
		self.output_list = [[char for char in line] for line in self.output_list]
		#生成的数据是一个list##[['\ufeff', '今天', '吃', '了', '吗'], ['你', '在', '干什么'], ['晚上', '下雨', '吗'], ['你', '叫', '什么', '名字'],
		#print(self.input_list)



	def _init_vocab(self):
		# 生成输入字典
		helper = []
		for line in self.input_list: helper += line#通过列表加法，就将所有句子整合成了一个。即helper是这样['\ufeff', '今天', '吃', '了', '吗', '你', '在', '干什么', '晚上', '下雨', '吗', '你', '叫',
		self.input_vocab = set(helper)
		self.id2inp = self.SOURCE_CODES + list(self.input_vocab)
		self.inp2id = {c:i for i,c in enumerate(self.id2inp)}#这里i是序号，c是值
		# 输出字典
		helper = []
		for line in self.output_list: helper += line
		self.output_vocab = set(helper)
		self.id2out = self.TARGET_CODES + list(self.output_vocab)
		self.out2id = {c:i for i,c in enumerate(self.id2out)}
		#print(self.out2id)#{'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3, '骂人': 4, '陈': 5, '≦': 6, ',': 7, '高': 8, '度': 9, 'и': 10, '切糕': 11, '御姐



	def _init_num_data(self):
		# 利用字典，映射数据
		self.en_inp_num_data = [[self.inp2id[en] for en in line] for line in self.input_list]
		#self.en_inp_num_data [[296, 81, 556, 150, 234], [167, 348, 147], [291, 371, 234], [167, 61, 499, 153], [64], [167, 233, 221, 499],
		self.de_inp_num = [[self.out2id['<GO>']] + [self.out2id[ch] for ch in line] for line in self.output_list]
		#为每一句话的前面加上'<GO>'所对应的字典数字，self.de_inp_num [[3, 457, 933, 239, 219], [3, 308, 551, 162, 655], [3, 780, 871, 611], [3, 308, 95, 157, 734, 610],

		self.de_out_num = [[self.out2id[ch] for ch in line] + [self.out2id['<EOS>']] for line in self.output_list]
		#为每一句话的后边加上'<EOS>'所对应的字典数字self.de_out_num  [[457, 933, 239, 219, 1], [308, 551, 162, 655, 1], [780, 871, 611, 1], [308, 95, 157, 734, 610, 1],
		#上边的数据都是一个二维list,一维（即最外维）指的句子的个数，二维指的句子包含的词汇。

	def generator(self, batch_size):
		batch_num = len(self.en_inp_num_data) // batch_size# 因此，len(self.en_inp_num_data),获取的是句子的个数。
		for i in range(batch_num):
			begin = i * batch_size
			end = begin + batch_size

			encoder_inputs = self.en_inp_num_data[begin:end]#取得0到end-1,即取得batch_size个句子
			decoder_inputs = self.de_inp_num[begin:end]
			decoder_targets = self.de_out_num[begin:end]

			encoder_lengths = [len(line) for line in encoder_inputs] #一个一维列表，表示encoder_inputs中每个句子的词汇个数，多少个句子， encoder_lengths就有多少个值
			decoder_lengths = [len(line) for line in decoder_inputs]
			encoder_max_length = max(encoder_lengths)#最长的那个句子的词汇数
			decoder_max_length = max(decoder_lengths)
			encoder_inputs = np.array([data + [self.inp2id['<PAD>']] * (encoder_max_length - len(data)) for data in encoder_inputs]).T
			#将所有句子变成和最长句子等长的向量，缺失值用'<PAD>'取代。最后转置。即每一列代表一个句子。形势是【max_length, batch_size】
			decoder_inputs = np.array([data + [self.out2id['<PAD>']] * (decoder_max_length - len(data)) for data in decoder_inputs]).T
			decoder_targets = np.array([data + [self.out2id['<PAD>']] * (decoder_max_length - len(data)) for data in decoder_targets]).T
			mask = decoder_targets > 0
			target_weights = mask.astype(np.int32)
			yield encoder_inputs, decoder_inputs, decoder_targets, target_weights, encoder_lengths, decoder_lengths


datav = GenData()
data_generator = datav.generator(32)
en_input, de_input, de_tg, tg_weight, en_len, de_len = next(data_generator)
print(de_tg.shape)

