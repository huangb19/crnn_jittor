import jittor as jt
from jittor.misc import CTCLoss
import numpy as np
import utils
from utils import BKTree
import dataset
from PIL import Image
from model import CRNN
import argparse
from Levenshtein import distance
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ALL')
parser.add_argument('--threshold', type=int, default=3)
parser.add_argument('--modelPath', type=str, default='./expr/crnn.pkl')
parser.add_argument('--datasetRoot', type=str, default='./test_dataset/')
parser.add_argument('--dictionaryRoot', type=str, default='./test_dataset/Hunspell_en.txt')
opt = parser.parse_args()
print(opt)

ds = opt.dataset
threshold = opt.threshold
model_path = opt.modelPath
root = opt.datasetRoot

if jt.has_cuda:
    print("using cuda")
    jt.flags.use_cuda = 1 
else:
    print("using cpu")

crnn = CRNN(1, 37, 256)
print('loading pretrained model from %s' % model_path)
crnn.load_state_dict(jt.load(model_path))

transformer = dataset.resizeNormalize((100, 32))
def readImage(img_path):
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    image = jt.array(np.expand_dims(image, axis=0))
    return image


with open(opt.dictionaryRoot, "r") as f:
    dict_list = [lex.lower() for lex in f.read().split()[:-1]]
bkTree = BKTree(dict_list)

converter = utils.strLabelConverter(alphabet='0123456789abcdefghijklmnopqrstuvwxyz')
criterion = CTCLoss(reduction="none")


def transcription(preds, lexs=[], useBKTree=False):
    preds_index = preds.data.argmax(2)
    preds_index = preds_index.transpose(1, 0)
    sim_pred = converter.decode(preds_index, raw=False)[0]
    if useBKTree:
        base_lexs = bkTree.find(sim_pred, threshold)
    else:
        base_lexs = [lex for lex in lexs if distance(sim_pred, lex) <= threshold]

    if len(base_lexs) == 0:
        return sim_pred
    else:
        text, length = converter.encode(base_lexs)
        loss = criterion(preds.repeat(1, len(base_lexs), 1), jt.array(text),
                         jt.array([preds.size(0)] * len(base_lexs)), jt.array(length)).data
        return base_lexs[loss.argmin()]


def test(ds, lex_type="None", lex_size=50):
    """
        ds:       in ["IIIT5K", "SVT", "IC03", "IC13"]
        lex_type: in ["None", "Build", "Full", "Read", "Dictionary"]
        lex_size: only used in "Build"
    """

    crnn.eval()
    base_dir = os.path.join(root, ds)
    with open(os.path.join(base_dir, "gt.txt"), "r") as f:
        lines = f.read().split("\n")
        while lines[-1] == "":
            lines = lines[:-1]

    base_lex = []
    if lex_type == "Full":
        for line in lines:
            gt = line.split()[1]
            base_lex.append(gt.lower())

    lex_map = {}
    if lex_type == "Read":
        with open(os.path.join(base_dir, "lex.txt"), "r") as f:
            lex_lines = f.read().split("\n")
        for line in lex_lines:
            if len(line) == 0: 
                break
            key, lex_50 = line.split()
            lex_map[key] = [lex.lower() for lex in lex_50.split(",")]

    tot = len(lines)
    acc = 0
    for i, line in enumerate(lines):
        image_path, gt = line.split()
        gt = gt.lower()

        image = readImage(os.path.join(base_dir, "image", image_path))
        preds = crnn(image)
        preds = jt.nn.log_softmax(preds, dim=2)

        if lex_type == "Build" and gt not in base_lex:
            cnt = 0
            base_lex = []
            while cnt < lex_size:
                base_lex.append(lines[(i + cnt) % tot].split()[1].lower())
                cnt += 1
        elif lex_type == "Read":
            base_lex = lex_map[image_path[:5]]

        pred_str = transcription(preds, base_lex, useBKTree=(lex_type == "Dictionary"))

        if lex_type == "Dictionary" and gt not in dict_list:
            tot -= 1

        # print(gt, pred_str)
        if gt == pred_str:
            acc += 1

    if lex_type == "Build":
        lex_type += str(lex_size)
    print("dataset:{}({}) {}/{} acc:{}".format(ds, lex_type, acc, tot, acc / tot))


if ds == "ALL" or ds == "IIIT5K":
    test("IIIT5K", "Build", 50)
    test("IIIT5K", "Build", 1000)
    test("IIIT5K", "None")

if ds == "ALL" or ds == "SVT":
    test("SVT", "Read")
    test("SVT", "None")

if ds == "ALL" or ds == "IC03":
    test("IC03", "Build", 50)
    test("IC03", "Full")
    test("IC03", "Dictionary")
    test("IC03", "None")

if ds == "ALL" or ds == "IC13":
    test("IC13", "None")


