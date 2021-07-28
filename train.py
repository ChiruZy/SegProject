import torch
import os
import argparse
import logging
from tqdm import tqdm
from nets import UNet, AttUNet, VGG_FCN, Res_FCN, UNet_CBAM, UNet_SE
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import dice_loss, miou, SimpleDataset, TransSet


def get_args():
    parser = argparse.ArgumentParser(description='Train the nets on different datasets')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-n', '--net', type=str, default='UNet_SE',
                        help='UNet AttUNet UNet_CBAM UNet_SE VGG_FPN Res_FPN')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-i', '--input_channels', type=int, default=3, help='input channels')
    parser.add_argument('-o', '--output_channels', type=int, default=1, help='output channels')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-p', '--pretrain', action="store_true", help='use pretrain(only VGG_FPN, Res_FPN)')
    parser.add_argument('-d', '--dataset_path', type=str, default='./datas/Glas', help='Dataset path')

    return parser.parse_args()


def get_net(name, **kwargs):
    net = eval(name)(kwargs)
    assert isinstance(net, nn.Module), "it's not a subclass of nn.Module"
    return net


def predict_label(pred):
    if args.output_channels == 1:
        res = torch.zeros_like(pred)
        res[pred > 0] = 1
        return res
    else:
        return torch.argmax(pred, dim=1, keepdim=True)


def train_net(net, args, device):
    # set hyperparameters
    weight_decay = 1e-4
    eta_min = 0

    # get dataset & dataloader
    train = SimpleDataset(args.dataset_path + '/train', output_channel=args.output_channels,
                          trans_set=TransSet(resize=(256, 384), color_jitter=(0.1, 0.1, 0.1, 0.1), flip_p=0.5))
    val = SimpleDataset(args.dataset_path + '/val', output_channel=args.output_channels,
                        trans_set=TransSet(resize=(256, 384)))
    test = SimpleDataset(args.dataset_path + '/test', output_channel=args.output_channels,
                         trans_set=TransSet(resize=(256, 384)))

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # set optim, loss func, scheduler
    optimizer = optim.SGD(list(net.parameters()), lr=args.learning_rate, weight_decay=weight_decay, momentum=0.9)
    criterion = nn.CrossEntropyLoss() if args.output_channels > 1 else nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train) // args.batch_size,
                                                     eta_min=eta_min)

    # set summary
    if not os.path.exists('runs'):
        os.mkdir('runs')

    # logging info
    logging.info(f'''\n    Starting training:
    Epochs:          {args.epochs}
    Batch size:      {args.batch_size}
    Learning rate:   {args.learning_rate}
    Training size:   {len(train)}
    Validation size: {len(val)}
    Testing size:    {len(test)}
    Device:          {device.type}''')

    max_iou = 0
    hparam_info = f'{args.net}-e{args.epochs}-bs{args.batch_size}-lr{args.learning_rate}'

    writer = SummaryWriter(log_dir=f'runs/{hparam_info}')
    writer.add_text('hparam', str({'net': args.net,
                                   'epoch': args.epochs,
                                   'batch size': args.batch_size,
                                   'learning rate': args.learning_rate,
                                   'weight decay': weight_decay,
                                   'lr scheduler': 'CosineAnnealingLR',
                                   'eta min': eta_min}))

    # start
    for epoch in range(args.epochs):

        # train
        with tqdm(total=len(train), desc=f'train: {epoch + 1}/{args.epochs}', unit='img', ascii=True) as pbar:
            epoch_loss, iou, n = 0, 0, 0
            net.train()
            for img, mask in train_loader:
                # forward
                img, mask = map(lambda x: x.to(device), [img, mask])
                mask_pre = net(img)
                loss = criterion(mask_pre, mask) + dice_loss(mask_pre, mask, args.output_channels)

                # backward
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                scheduler.step()

                # calc loss & iou
                epoch_loss += loss.item()
                iou += miou(predict_label(mask_pre.detach()), mask, args.output_channels)
                n += 1

                # pbar update
                pbar.set_postfix(**{'mean loss:': epoch_loss / n, 'm_IoU': iou / n})
                pbar.update(img.shape[0])

        writer.add_scalar('Loss/train', epoch_loss / n, epoch)
        writer.add_scalar('m_IoU/train', iou / n, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # val
        if not val:
            continue
        with tqdm(total=len(val), desc=f'val:   {epoch + 1}/{args.epochs}', unit='img', ascii=True) as pbar:
            net.eval()
            iou, n = 0, 0

            with torch.no_grad():
                for idx, (img, mask) in enumerate(val_loader):
                    img, mask = map(lambda x: x.to(device), [img, mask])
                    mask_pred = predict_label(net(img).detach())

                    iou += miou(mask_pred, mask, args.output_channels)
                    n += 1

                    pbar.set_postfix(**{'m_IoU:': iou / n})
                    pbar.update(img.shape[0])

                    if idx == 0:
                        writer.add_images('test/images', img, epoch)
                        writer.add_images('test/masks/true', mask, epoch)
                        writer.add_images('test/masks/pred', mask_pred, epoch)

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
            net.load_state_dict(torch.load(f'checkpoint/{hparam_info}.pt'))
            net.eval()

            with torch.no_grad():
                for img, mask in test_loader:
                    img, mask = map(lambda x: x.to(device), [img, mask])
                    mask_pred = predict_label(net(img).detach())

                    iou += miou(mask_pred, mask, args.output_channels)
                    n += 1

                    pbar.set_postfix(**{'m_IoU:': iou / n})
                    pbar.update(img.shape[0])

                    writer.add_images('images', img)
                    writer.add_images('masks/true', mask)
                    writer.add_images('masks/pred', mask_pred)

            writer.add_text('test m_IoU', str(iou / n))
        logging.info(f'Test m_IoU is {iou / n}.')

    # end train
    writer.close()


if __name__ == '__main__':
    # parser args
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    net = get_net(args.net, in_c=args.input_channels, out_c=args.output_channels, pretrain=args.pretrain).to(device=device)

    # logging setting
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
    logging.info(f'\n\tNetwork: {args.net}\n'
                 f'\t{args.input_channels} input channels\n'
                 f'\t{args.output_channels} output channels (classes)\n')

    # train
    train_net(net, args, device)
