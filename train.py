import torch
import os
import argparse
import logging
from tqdm import tqdm
from nets import UNet, AttUNet, VGG_FCN, Res_FCN, UNet_CBAM, UNet_SE, PNet, VGG_BN_FCN
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import dice_loss, miou, SimpleDataset, TransSet
import random
import numpy as np


def set_random_seed(seed=1):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='Train the nets on different datasets')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('-n', '--net', type=str, default='PNet',
                        help='UNet AttUNet UNet_CBAM UNet_SE VGG_FCN Res_FCN VGG_BN_FCN')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-i', '--input_channels', type=int, default=3, help='input channels')
    parser.add_argument('-o', '--output_channels', type=int, default=1, help='output channels')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.03, help='Learning rate')
    parser.add_argument('-p', '--pretrain', action="store_false", help='use pretrain(only for net include VGG or Res)')
    parser.add_argument('-d', '--dataset_path', type=str, default='./datas/Glas', help='Dataset path')
    parser.add_argument('-a', '--gradient_accumulation', type=int, default=16, help='Gradient accumulation')
    parser.add_argument('-s', '--save', action="store_false", help='save train data')

    args = vars(parser.parse_args())
    args['seed'] = 1
    args['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return args


def get_net(name, **kwargs):
    net = eval(name)(kwargs)
    assert isinstance(net, nn.Module), "it's not a subclass of nn.Module"
    return net


def predict_label(pred):
    if args['output_channels'] == 1:
        res = torch.zeros_like(pred)
        res[pred > 0] = 1
        return res
    else:
        return torch.argmax(pred, dim=1, keepdim=True)


def train_net(net, args):
    ep = args['epoch']
    tb = args["batch_size"] if args["gradient_accumulation"] == 0 else args["gradient_accumulation"]
    oc = args['output_channels']
    dp = args["dataset_path"]
    bs = args['batch_size']
    lr = args['learning_rate']
    wd = 1e-4
    em = None

    net_name = args['net']
    device = args['device']
    save = args['save']



    # get dataset & dataloader
    train = SimpleDataset(dp + '/train', output_channel=oc,
                          trans_set=TransSet(resize=(224, 224), color_jitter=(0.1, 0.1, 0.1, 0.1), flip_p=0.5))
    val = SimpleDataset(dp + '/val', output_channel=oc, trans_set=TransSet(resize=(224, 224)))
    test = SimpleDataset(dp + '/test', output_channel=oc, trans_set=TransSet(resize=(224, 224)))

    train_loader = DataLoader(train, batch_size=bs, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=bs, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, pin_memory=True)

    # set optim, loss func, scheduler
    optimizer = optim.SGD(list(net.parameters()), lr=lr, weight_decay=wd, momentum=0.9)
    criterion = nn.CrossEntropyLoss() if oc > 1 else nn.BCEWithLogitsLoss()
    scheduler = None
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ep * len(train) // tb, eta_min=em)

    # set summary
    if not os.path.exists('runs'):
        os.mkdir('runs')

    # logging info
    logging.info(f'''\n    Starting training:
    Epochs:          {ep}
    Batch size:      {tb}
    Learning rate:   {lr}
    Training size:   {len(train)}
    Validation size: {len(val)}
    Testing size:    {len(test)}
    Device:          {device.type}''')

    max_iou = 0
    hparam_info = f'{net_name}-e{ep}-bs{tb}-lr{lr}-seed_{args["seed"]}'

    writer = None
    if save:
        writer = SummaryWriter(log_dir=f'runs/{hparam_info}')
        writer.add_text('hparam', str({'net': net_name,
                                       'epoch': ep,
                                       'batch size': bs,
                                       'learning rate': lr,
                                       'weight decay': wd,
                                       'lr scheduler': scheduler.__class__.__name__,
                                       'eta min': em}))

    # start
    optimizer.zero_grad()
    for epoch in range(ep):

        with tqdm(total=len(train), desc=f'train: {epoch + 1}/{ep}', unit='img', ascii=True) as pbar:
            epoch_loss, iou, n = 0, 0, 0
            net.train()
            for idx, (img, mask) in enumerate(train_loader):
                # forward
                img, mask = map(lambda x: x.to(device), [img, mask])
                mask_pre = net(img)
                loss = criterion(mask_pre, mask) + dice_loss(mask_pre, mask, oc)

                # backward
                loss.backward()

                # loss accumulation
                if bs * (idx + 1) % tb == 0 or len(train_loader) == idx + 1:
                    # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler:
                        scheduler.step()

                # calc loss & iou
                epoch_loss += loss.item()
                iou += miou(predict_label(mask_pre.detach()), mask, oc)
                n += 1

                # pbar update
                pbar.set_postfix(**{'mean loss:': epoch_loss / n / bs, 'm_IoU': iou / n})
                pbar.update(img.shape[0])

        if writer:
            writer.add_scalar('Loss/train', epoch_loss / n / bs, epoch)
            writer.add_scalar('m_IoU/train', iou / n, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # val
        if not val:
            continue
        with tqdm(total=len(val), desc=f'val:   {epoch + 1}/{ep}', unit='img', ascii=True) as pbar:
            net.eval()
            iou, n = 0, 0

            with torch.no_grad():
                for idx, (img, mask) in enumerate(val_loader):
                    img, mask = map(lambda x: x.to(device), [img, mask])
                    mask_pred = predict_label(net(img).detach())

                    iou += miou(mask_pred, mask, oc)
                    n += 1

                    pbar.set_postfix(**{'m_IoU:': iou / n})
                    pbar.update(img.shape[0])

                    if writer and idx == 0:
                        writer.add_images('val/images', img, epoch)
                        writer.add_images('val/masks/true', mask, epoch)
                        writer.add_images('val/masks/pred', mask_pred, epoch)

            if writer:
                writer.add_scalar('m_IoU/val', iou / n, epoch)
                if iou / n > max_iou:
                    max_iou = iou / n
                    if not os.path.exists('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(net.state_dict(), f'checkpoint/{hparam_info}.pt')
                    pbar.set_postfix(**{'m_IoU:': f'{iou / n:.3f}, params saved!'})

    logging.info(f'Training is over, the max m_IoU is {max_iou}.')

    # test
    if test:
        iou, n = 0, 0
        with tqdm(total=len(test), desc=f'test: ', unit='img', ascii=True) as pbar:

            if save:
                net.load_state_dict(torch.load(f'checkpoint/{hparam_info}.pt'))

            net.eval()
            with torch.no_grad():
                for img, mask in test_loader:
                    img, mask = map(lambda x: x.to(device), [img, mask])
                    mask_pred = predict_label(net(img).detach())

                    iou += miou(mask_pred, mask, oc)
                    n += 1

                    pbar.set_postfix(**{'m_IoU:': iou / n})
                    pbar.update(img.shape[0])

                    if writer:
                        writer.add_images('images', img)
                        writer.add_images('test/masks/true', mask)
                        writer.add_images('test/masks/pred', mask_pred)
            if writer:
                writer.add_text('test m_IoU', str(iou / n))
        logging.info(f'Test m_IoU is {iou / n}.')

    # end train
    if writer:
        writer.close()


if __name__ == '__main__':
    # parser args
    args = get_args()
    set_random_seed(args['seed'])
    net = get_net(args['net'], in_c=args['input_channels'], out_c=args['output_channels'],
                  pretrain=args['pretrain']).to(device=args['device'])

    # logging setting
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
    logging.info(f'\n\tNetwork: {args["net"]}\n'
                 f'\t{args["input_channels"]} input channels\n'
                 f'\t{args["output_channels"]} output channels (classes)\n')

    # train
    train_net(net, args)
