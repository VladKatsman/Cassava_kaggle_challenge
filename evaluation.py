import torch
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from train import init_device_pytorch, create_model, AverageMeter, accuracy
from dataset.dataset import dataloader
from dataset.augmentations import get_inference_transforms
from data_preprocessing.train_val_split import split_train_val_5_fold


def eval_one_image(path, model, device):

    image = np.array(Image.open(path))
    image = preprocess(image=image)['image']
    image = torch.unsqueeze(image, dim=0)
    image = image.to(device)
    with torch.no_grad():
        prediction = model(image)
        print(prediction.argmax())


def eval_one_folder(pathes, model, device):

    for path in pathes:
        im = Image.open(path)
        eval_one_image(path, model, device)
        pass


def eval_test_set(dataset, model, device):

    run_val_acc = AverageMeter('val_acc')
    pbar = tqdm(dataset, total=len(dataset))
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            acc_1 = accuracy(predictions, labels)
            run_val_acc.update(acc_1[0].item(), images.size(0))
            pbar.set_description('Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(top1=run_val_acc))
    print(run_val_acc.avg)


if __name__ == '__main__':

    # global parameters
    ROOT = '/home/noteme/PycharmProjects/comp/data/cassava-disease'
    WEIGHTS = '//home/noteme/PycharmProjects/comp/data/cassava-disease/experiment/model_best.pth'
    SEED = 1337

    # test
    test_image = '/home/noteme/PycharmProjects/comp/data/2216849948.jpg'

    # restore original test splits
    folds, images_list = split_train_val_5_fold(root=ROOT, seed=SEED)

    # initiate gpus indexes
    gpus, device = init_device_pytorch()

    # create model
    model = create_model()
    model = model.to(device)

    weights = torch.load(WEIGHTS)
    model.load_state_dict(weights['state_dict'])

    # set model to evaluation
    model.eval()

    # initiate image preprocessor
    preprocess = get_inference_transforms()

    # initiate dataset
    dataset_val = dataloader(split=folds[0]['train'], images_list=images_list, batch_size=32, train=False)

    # result = eval_one_image(test_image, model, device)
    # result = eval_test_set(dataset_val, model, device)
    results = eval_one_folder(path_to_healty, model, device)
