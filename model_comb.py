import jittor as jt
import jittor.contrib
from jittor import nn, Module
from jittor.misc import CTCLoss
from time import time


# ----- sub-module to build other modules ----- #

class BidirectionalLSTM(Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def execute(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]  TODO:这里用了pytorch版本的维数，看起来是有问题的
        output = output.view(T, b, -1)

        return output

def cross_entropy_loss(preds, label):
    loss = preds.exp()  # e^x
    loss = (loss / loss.sum(dim=1, keepdims=True)).log()  # log probability
    loss = label * loss
    return -loss.sum(dim=1)

def dense_loss(preds, label, length, batch_size):
    preds = preds.permute(1, 0, 2)  # batch_size * length * nclass
    preds_reshape = preds.reshape(-1, preds.shape[2])
    label_reshape = jt.float32(label).reshape(-1)

    # label 转成 one-hot 表示
    label_onehot = label_reshape[...,None] == jt.index(list(label_reshape.shape)+[preds.shape[2],])[-1]

    # 计算 loss
    #loss = jt.nn.cross_entropy_loss(preds_reshape, jt.array(label_reshape), reduction='none')
    loss = cross_entropy_loss(preds_reshape, label_onehot)
    loss = loss.reshape((preds.shape[0], -1))
    #print(loss)

    # 构造 mask
    mask = (label + jt.contrib.concat([jt.zeros((label.shape[0], 1)), label[:, :-1]], dim=1)) > 0  # batch_size * length
    loss = loss * mask
    loss = loss.sum() / preds.shape[0]
    return loss

class ResUnit(Module):
    def __init__(self, channel_in, channel_out, leakyRelu=False):
        super(ResUnit, self).__init__()

        # full calculation
        self.calc = nn.Sequential(
            nn.BatchNorm2d(channel_in),
            nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
            nn.Conv2d(channel_in, channel_out, 3, 1, 1),
            nn.BatchNorm2d(channel_out),
            nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

        # short cut
        self.direct_addition = channel_in == channel_out
        if not self.direct_addition:
            self.short_cut = nn.Conv2d(channel_in, channel_out, 1, 1, 0)

    def execute(self, input):
        if self.direct_addition:
            return input + self.calc(input)
        else:
            return self.short_cut(input) + self.calc(input)


# ----- direct candidates for cnn and rnn parts ----- #

class CNNSlideWindow(Module):
    def __init__(self, nc, leakyRelu=False):
        super(CNNSlideWindow, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2] # kernel_size
        ps = [1, 1, 1, 1, 1, 1, 0] # padding
        ss = [1, 1, 1, 1, 1, 1, 1] # stride
        nm = [64, 128, 256, 256, 512, 512, 512] # channel

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU())

        # 1x32x100
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x50
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x25
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x26
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x27
        convRelu(6, True)  # 512x1x26

        self.cnn = cnn

    def execute(self, input):
        conv = self.cnn(input)
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # [width, batch, feature = channel * height] = [26, batch, 512]
        return conv


class CNNGlobal(Module):
    def __init__(self, nc, leakyRelu=False):
        super(CNNGlobal, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2] # kernel_size
        ps = [1, 1, 1, 1, 1, 1, 0] # padding
        ss = [1, 1, 1, 1, 1, 1, 1] # stride
        nm = [64, 128, 256, 256, 512, 512, 512] # channel

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU())

        # 1x32x100
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x50
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x25
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x26
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x27
        convRelu(6, True)  # 512x1x26

        self.cnn = cnn

    def execute(self, input):
        conv = self.cnn(input)
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel, height * width)
        conv = conv.permute(1, 0, 2)  # [channel, batch, feature_length = width * height] = [512, batch, 26]
        return conv


class CNNRes(Module):
    def __init__(self, nc, leakyRelu=False):
        super(CNNRes, self).__init__()

        self.res = nn.Sequential(
            # nc * 32 * 100 -> 64 * 16 * 50
            nn.Conv2d(nc, 64, 3, 1, 1),
            ResUnit(64, 64, leakyRelu),
            ResUnit(64, 64, leakyRelu),
            nn.MaxPool2d(2, 2),

            # 64 * 16 * 50 -> 128 * 8 * 25
            ResUnit(64, 128, leakyRelu),
            ResUnit(128, 128, leakyRelu),
            ResUnit(128, 128, leakyRelu),
            nn.MaxPool2d(2, 2),

            # 128 * 8 * 25 -> 256 * 4 * 26
            ResUnit(128, 256, leakyRelu),
            ResUnit(256, 256, leakyRelu),
            ResUnit(256, 256, leakyRelu),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            # 256 * 4 * 26 -> 512 * 2 * 27
            ResUnit(256, 512, leakyRelu),
            ResUnit(512, 512, leakyRelu),
            ResUnit(512, 512, leakyRelu),
            nn.MaxPool2d((2, 2), (2,1), (0, 1)),

            # 512 * 2 * 27 -> 512 * 1 * 26
            nn.Conv2d(512, 512, 2, 1, 0)
        )

    def execute(self, input):
        conv = self.res(input)
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # [width, batch, feature = channel * height] = [26, batch, 512]
        return conv


class RNNLstmCtc(Module):
    def __init__(self, feature_length, nclass, nh):
        super(RNNLstmCtc, self).__init__()

        self.rnn = nn.Sequential(
            BidirectionalLSTM(feature_length, nh, nh),    # BiLSTM (512 -> nh) + Linear (2 * nh -> nh)
            BidirectionalLSTM(nh, nh, nclass)) # BiLSTM (nh -> nh)  + Linear (2 * nh -> nclass)

        self.loss = CTCLoss()

        self.output_dense = False  # 输出是不紧凑的 (abb--cc-)

    def execute(self, input):
        return self.rnn(input[0])

    def calc_loss(self, preds, label, length, batch_size):
        preds = jt.nn.log_softmax(preds, dim=2)
        loss = self.loss(preds, jt.array(label), jt.array([preds.size(0)] * batch_size), jt.array(length)) / batch_size
        return loss


class RNNEncoderDecoder(Module):
    def __init__(self, feature_length, nclass):
        super(RNNEncoderDecoder, self).__init__()

        self.reoptimize = nn.LSTM(feature_length, int(feature_length / 2), bidirectional=True)

        self.cell = nn.LSTMCell(feature_length, feature_length)

        self.linear = nn.Linear(feature_length, nclass)

        self.embedding = nn.Linear(1, feature_length)

        self.eos = jt.zeros(nclass)
        self.eos[0] = 1.
        self.eos = self.eos.reshape((1, -1))

        self.nclass = nclass

        self.output_dense = True  # 输出是紧凑的 (abc)

    def execute(self, input):
        feature, _ = self.reoptimize(input[0])  # feature_num * batch_size * feature_length

        # encode
        hx, cx = self.cell(feature[0])
        for i in range(1, feature.shape[0]):
            hx, cx = self.cell(feature[i], (hx, cx))

        # decode
        if self.is_training():
            # input: batch_size * length
            # label: length * batch_size
            label = jt.float32(
                jt.contrib.concat([jt.zeros((input[1].shape[0], 1)), input[1][:, :-1]], dim=1).transpose(1, 0)
            )
            # teacher forcing
            outputs = []
            for i in range(label.shape[0]):
                x = self.embedding(label[i].reshape([-1, 1]))
                hx, cx = self.cell(x, (hx, cx))
                y = self.linear(hx)
                outputs.append(y.reshape([1, y.shape[0], -1]))
            outputs = jt.contrib.concat(outputs)
            return outputs
        else:
            outputs = []
            mask = jt.ones((feature.shape[1], self.nclass))
            x = self.embedding(jt.zeros((input[1].shape[0], 1)))
            #print('length:', input[1].shape)  # batch_size * length
            for i in range(input[1].shape[1]):
                hx, cx = self.cell(x, (hx, cx))
                y = self.linear(hx)  # batch_size * nclass
                y = y * mask + self.eos * (1 - mask)
                outputs.append(y.reshape([1, y.shape[0], -1]))
                # 将输出转换成确定的字母，作为下一轮的输入
                preds_index = nn.log_softmax(y, dim=1).argmax(1)[0]
                preds_index = preds_index.reshape([-1, 1])
                mask = mask * (preds_index > 0)
                x = self.embedding(preds_index)
            outputs = jt.contrib.concat(outputs)
            return outputs

    def calc_loss(self, preds, label, length, batch_size):
        return dense_loss(preds, label, length, batch_size)


class RNNAttention(Module):
    def __init__(self, feature_length, nclass):
        self.reoptimize = nn.LSTM(feature_length, int(feature_length / 2), bidirectional=True)

        self.cell = nn.LSTMCell(feature_length, feature_length)

        self.to_letter = nn.Linear(feature_length, nclass)

        self.attention_state = nn.Linear(feature_length, feature_length)
        self.attention_f = nn.Linear(feature_length, feature_length)
        self.attention_weight = nn.Linear(feature_length, 1)

        self.eos = jt.zeros(nclass)
        self.eos[0] = 1.
        self.eos = self.eos.reshape((1, -1))

        self.nclass = nclass

        self.output_dense = True  # 输出是紧凑的 (abc)

    def execute(self, input):
        feature, _ = self.reoptimize(input[0])  # feature_num * batch_size * feature_length

        # encode
        hx, cx = self.cell(feature[0])
        for i in range(1, feature.shape[0]):
            hx, cx = self.cell(feature[i], (hx, cx))

        # decode
        outputs = []
        mask = jt.ones((feature.shape[1], self.nclass))

        for i in range(input[1].shape[1]):
            # calculate current input
            # hx: batch_size * feature_length
            # feature: feature_num * batch_size * feature_length
            U = self.attention_state(hx)
            V = self.attention_f(feature)
            W = U.reshape((1, U.shape[0], -1)) + V
            #print(V.shape)
            weights = self.attention_weight(W)  # feature_num * batch_size * 1
            weights = weights.reshape(weights.shape[0], -1)  # feature_num * batch_size
            weights = nn.softmax(weights, dim=0)
            # x: batch_size * feature_length
            x = weights.reshape(weights.shape[0], -1, 1) * feature  # feature_num * batch_size * feature_length
            x = x.sum(dim=0)  # batch_size * feature_length

            # step forward
            hx, cx = self.cell(x, (hx, cx))
            y = self.to_letter(hx)  # batch_size * nclass
            y = y * mask + self.eos * (1 - mask)
            outputs.append(y.reshape([1, y.shape[0], -1]))

            # generate mask
            preds_index = nn.log_softmax(y, dim=1).argmax(1)[0]
            preds_index = preds_index.reshape([-1, 1])
            mask = mask * (preds_index > 0)

        outputs = jt.contrib.concat(outputs)
        return outputs

    def calc_loss(self, preds, label, length, batch_size):
        return dense_loss(preds, label, length, batch_size)


# ----- top module ----- #

class CRNN(Module):

    def __init__(self, cnn_type, rnn_type, para):
        super(CRNN, self).__init__()

        if cnn_type == 'CNNSlideWindow':
            self.cnn = CNNSlideWindow(para['nc'], para['leakyRelu'])
            para['feature_length'] = 512
        elif cnn_type == 'CNNGlobal':
            self.cnn = CNNGlobal(para['nc'], para['leakyRelu'])
            para['feature_length'] = 26
        elif cnn_type == 'CNNRes':
            self.cnn = CNNRes(para['nc'], para['leakyRelu'])
            para['feature_length'] = 512

        if rnn_type == 'RNNLstmCtc':
            self.rnn = RNNLstmCtc(para['feature_length'], para['nclass'], para['nh'])
        elif rnn_type == 'RNNEncoderDecoder':
            self.rnn = RNNEncoderDecoder(para['feature_length'], para['nclass'])
        elif rnn_type == 'RNNAttention':
            self.rnn = RNNAttention(para['feature_length'], para['nclass'])


    def execute(self, input):
        # conv features
        conv = self.cnn(input[0])

        # rnn features
        output = self.rnn((conv, input[1]))
        return output

    def calc_loss(self, preds, label, length, batch_size):
        return self.rnn.calc_loss(preds, label, length, batch_size)

    def is_output_dense(self):
        return self.rnn.output_dense
