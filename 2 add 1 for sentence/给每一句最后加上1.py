import numpy as np
import tensorflow as tf
import functools

#参数设置
PAD = 0
EOS = 1

# embedding parameters
vocab_size = 10  #词典数量 约等于最后输出的数据库
input_embedding_size = 20  #embedding输入层大小 词的向量化的输入数据

# network parameters
encoder_hidden_units = 20
decoder_hidden_units = 20

# training parameters
batch_size = 100
max_batches = 3001 # 训练需要执行的batch的个数
batches_in_epoch = 1000


def gen_batch(inputs, max_seq_length=None):
    '''
    将inputs转换为numpy数组并将所有sequence用0填充到等长。
    参数
    -------
    inputs : (batch_size,seq_len)
    '''
    sequence_lengths = [len(seq) for seq in inputs] #用于获取每个序列的长度
    batch_size = len(inputs) #用于保存矩阵的行数，也就是batch——size
    if max_seq_length is None:
        max_seq_length = max(sequence_lengths)  #使用输入序列的最长序列长度作为长度限制
    # inputs对应的numpy数组，其中batch_size作为axis=0
    inputs_batch_major = np.zeros(shape=[batch_size, max_seq_length], dtype=np.int32) #初始为0随机化数组
    # seq:(seq_len,input_num)
    for i, seq in enumerate(inputs): #enumerate常用于循环中，用于将下标与数据分离 先分离每一行
        # element:input_num
        for j, element in enumerate(seq): #用于分离第二维的数据
            inputs_batch_major[i, j] = element #传入每一个数据
    # 将seq_len作为axis=0 因为后面需要time——step为第一维
    inputs_seq_major = inputs_batch_major.swapaxes(0, 1)#这里变换了维度，将【batch_size, max_lengths】变为【 max_lengths，batch_size】
    return inputs_seq_major, max_seq_length #返回随机数列与序列长度,inputs_seq_major为【 max_lengths，batch_size】

def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    '''
    随机产生batch_size个sequences，
    其中sequences的长度介于[length_from,length_to],
    sequences中的值介于[vocab_lower,vocab_upper]
    '''
    if length_from > length_to: #错误异常判断
        raise ValueError('length_from > length_to')
    def random_length():
        '''
        随机产生介于[length_from,length_to]的整数
        '''
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()  #生成batchsize组数据
            for _ in range(batch_size)
        ]#生成【100，seq】

def next_feed(batches):
    '''
    用于train时，为sess产生feed_dict
    '''
    batch = next(batches) # 产生当前batch的数据
    encoder_inputs_, _ = gen_batch(batch) # 将该batch的数据处理为encoder期望的形式
    # decoder_inputs_是在原始sequence前拼接EOS
    decoder_inputs_, _ = gen_batch(
        [[EOS] + (sequence) for sequence in batch]
    )

    # decoder_targets_是在原始sequence后拼接EOS
    decoder_targets_, _ = gen_batch(
        [(sequence) + [EOS] for sequence in batch]
    )

    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

# encoder_inputs:(batch_size,seq_len)
encoder_inputs = tf.placeholder(shape=[None,None], dtype=tf.int32, name='encoder_inputs')
# decoder_inputs:(batch_size,seq_len+1)
decoder_inputs = tf.placeholder(shape=[None,None], dtype=tf.int32, name='decoder_inputs')
# decoder_targets:(batch_size,seq_len+1)
decoder_targets = tf.placeholder(shape=[None,None], dtype=tf.int32, name='decoder_targets')
# embedding  用于将词的ID转化为向量用于输入到RNN中
lookup_table = tf.Variable(tf.random_normal([vocab_size,input_embedding_size],-1.,1.),dtype=tf.float32)
# eocoder_inputs_embedded:(batch_size,seq_len,input_embedding_size)
encoder_inputs_embedded = tf.nn.embedding_lookup(lookup_table,encoder_inputs)#【max_lengths,batch_size, 20】
decoder_inputs_embedded = tf.nn.embedding_lookup(lookup_table,decoder_inputs)#【max_lengths,batch_size, 20】
"""
函数：
tf.nn.embedding_lookup(

               params,

               ids,

               partition_strategy='mod',

               name=None,

              validate_indices=True,

              max_norm=None

)

参数说明：
params: 表示完整的嵌入张量，或者除了第一维度之外具有相同形状的P个张量的列表，表示经分割的嵌入张量
ids: 一个类型为int32或int64的Tensor，包含要在params中查找的id
partition_strategy: 指定分区策略的字符串，如果len（params）> 1，则相关。当前支持“div”和“mod”。 默认为“mod”
name: 操作名称（可选）
validate_indices:  是否验证收集索引
max_norm: 如果不是None，嵌入值将被l2归一化为max_norm的值

tf.nn.embedding_lookup()函数的用法主要是选取一个张量里面索引对应的元素
tf.nn.embedding_lookup(tensor,id)：即tensor就是输入的张量，id 就是张量对应的索引
tf.nn.embedding_lookup()就是根据input_ids中的id，寻找embeddings中的第id行。比如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，组成一个tensor返回
embedding_lookup不是简单的查表，id对应的向量是可以训练的，训练参数个数应该是 category num*embedding size，也就是说lookup是一种全连接层

"""
# 定义encoder
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                         encoder_inputs_embedded,
                                                         dtype=tf.float32,
                                                         time_major=True) # time_major表示输入的axis=0轴为time(或者是seq_len)
del encoder_outputs # 将encoder的outputs丢弃，只将final_state传递给decoder

# 定义decoder
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs,decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                       decoder_inputs_embedded,
                                                       initial_state=encoder_final_state,
                                                       dtype=tf.float32,
                                                       time_major=True,
                                                       scope="plain_decoder")#【max_lengths,batch_size, 20】

# decoder_outputs是一个Tensor
#tf.contrib.layers.linear() = tf.contrib.layers.fully_connected()
decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size) #这里变成了【batch_size, mac_length, vocab_size】，
                                                                        #这一步，相当于先将【 mac_length,batch_size, input_embedding_size】
                                                                        #相当于另一个项目里的【batch_size*mac_length，input_embedding_size】，然后乘权重和偏置。最后变为
                                                                        #【batch_size, mac_length, vocab_size】
decoder_prediction = tf.argmax(decoder_logits, 2)
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),# mask = decoder_targets > 0
                                                                            # target_weights = mask.astype(np.int32)
    logits=decoder_logits)
        
loss = tf.reduce_mean(stepwise_cross_entropy)
#损失值
train_op = tf.train.AdamOptimizer().minimize(loss)
#Adam优化器

batches = random_sequences(length_from=3, length_to=8,
                           vocab_lower=2, vocab_upper=10,
                           batch_size=batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    loss_track = []
    for batch in range(max_batches):
        fd = functools.partial(next_feed, batches)() #每次循环产生一个batch的数据
        _, l = sess.run([train_op, loss], fd) #通过梯度下降训练，fd为字典
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break