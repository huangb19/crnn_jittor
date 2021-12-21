import jittor as jt
import numpy as np
import utils
import dataset
from PIL import Image
from model import CRNN

if jt.has_cuda:
    print("using cuda")
    jt.flags.use_cuda = 1 
else:
    print("using cpu")


model_path = './expr/crnn.pkl'
img_path = './images/demo.png'
# img_path = './images/demo2.jpg'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
image = jt.array(np.expand_dims(image, axis=0))

crnn = CRNN(1, 37, 256)
print('loading pretrained model from %s' % model_path)
crnn.load_state_dict(jt.load(model_path))

crnn.eval()
preds = crnn(image)
preds = jt.nn.log_softmax(preds, dim=2)
preds_index = preds.data.argmax(2)
preds_index = preds_index.transpose(1, 0)

converter = utils.strLabelConverter(alphabet)
raw_pred = converter.decode(preds_index, raw=True)
sim_pred = converter.decode(preds_index, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
