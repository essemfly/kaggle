import torch
import cv2
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from sklearn.metrics import cohen_kappa_score
from .preprocess import circle_crop_v2
from .contrast_enhance import clahe_channel, subtract_local_mean
from .scorer import OptimizedRounder


# Set seed for all
def seed_everything(seed=1882):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

PATH = Path('./')
IMG_SIZE = 512
DEVICE = 'cuda:0'


def _load_format(path, _convert_mode, _after_open) -> Image:
    image = circle_crop_v2(path)
    image = clahe_channel(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return Image(pil2tensor(image, np.float32).div_(255))  # return fastai Image format


def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'), device=DEVICE)


def main(model_name):
    df = pd.read_csv(PATH / 'train.csv')
    vision.data.open_image = _load_format
    src = (
        ImageList.from_df(df, PATH, folder='train_images', suffix='.png')
            .split_by_rand_pct(0.2)
            .label_from_df(cols='diagnosis', label_cls=FloatList)
    )
    tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.3, max_warp=0.0, max_lighting=0.2)
    data = (
        src.transform(tfms, size=128)
            .databunch()
            .normalize(imagenet_stats)
    )

    learn = cnn_learner(data, base_arch=models.resnet50, metrics=[quadratic_kappa], model_dir='./models',
                        pretrained=True)

    learn.lr_find()
    learn.recorder.plot()
    lr = 1e-2
    learn.fit_one_cycle(3, lr)

    learn.data = data = (
        src.transform(tfms, size=256)
            .databunch()
            .normalize(imagenet_stats)
    )

    learn.lr_find()
    learn.recorder.plot()
    lr = 1e-2
    learn.fit_one_cycle(3, lr)

    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot()
    learn.fit_one_cycle(5, slice(1e-6, 1e-3),
                        callbacks=[
                            SaveModelCallback(learn, monitor='quadratic_kappa', mode='max'),
                            ShowGraph(learn),
                            EarlyStoppingCallback(learn, min_delta=1e-6, patience=3),
                        ])
    torch.save(learn.model, './models/' + model_name)

    valid_preds = learn.get_preds(ds_type=DatasetType.Valid)
    optR = OptimizedRounder()
    optR.fit(valid_preds[0], valid_preds[1])
    coefficients = optR.coefficients()

    return coefficients
