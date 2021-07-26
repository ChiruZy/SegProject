import torch
import argparse
import logging
from tqdm import tqdm
from nets import UNet, AttUNet
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from utils import dice_loss, miou, SimpleDataset, TransSet


def get_args():
    parser = argparse.ArgumentParser(description='Train the nets on different datasets')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-n', '--net', type=str, default='AttUNet', help='Net')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-i', '--input_channels', type=int, default=3, help='input channels')
    parser.add_argument('-o', '--output_channels', type=int, default=1, help='output channels')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-d', '--dataset_path', type=str, default='./datas/Glas', help='Dataset path')
    parser.add_argument('-v', '--validation', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


def get_net(name, in_c, out_c):
    net = eval(f'{name}({in_c}, {out_c})')
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
    dataset = SimpleDataset(args.dataset_path, output_channel=args.output_channels, trans_set=TransSet(resize=(256, 400)))
    n_val = int(len(dataset) * (args.validation / 100))
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    val.trans_set = None

    optimizer = optim.SGD(list(net.parameters()), lr=args.learning_rate, weight_decay=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss() if args.output_channels > 1 else nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * n_train // args.batch_size, eta_min=1e-5)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{args.learning_rate}_BS_{args.batch_size}')

    logging.info(f'''    Starting training:
    Epochs:          {args.epochs}
    Batch size:      {args.batch_size}
    Learning rate:   {args.learning_rate}
    Training size:   {n_train}
    Validation size: {n_val}
    Device:          {device.type}''')

    for epoch in range(args.epochs):
        epoch_loss, iou, n = 0, 0, 0

        with tqdm(total=n_train, desc=f'epoch: {epoch+1}/{args.epochs}', unit='img') as pbar:
            net.train()
            for img, mask in train_loader:
                img, mask = map(lambda x: x.to(device), [img, mask])
                mask_pre = net(img)

                loss = criterion(mask_pre, mask) + dice_loss(mask_pre, mask, args.output_channels)
                epoch_loss += loss.item()
                iou += miou(predict_label(mask_pre.detach()), mask, args.output_channels)
                n += 1

                pbar.set_postfix(**{'mean loss:': epoch_loss/n, 'm_IoU': iou/n})


                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                pbar.update(img.shape[0])

        writer.add_scalar('Loss/train', epoch_loss / n, epoch)
        writer.add_scalar('m_IoU/train', iou / n, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        iou, n = 0, 0

        with tqdm(total=n_val, desc=f'val:   {epoch+1}/{args.epochs}', unit='img') as pbar:
            net.eval()
            with torch.no_grad():
                for idx, (img, mask) in enumerate(val_loader):
                    img, mask = map(lambda x: x.to(device), [img, mask])
                    mask_pred = predict_label(net(img).detach())

                    iou += miou(mask_pred, mask, args.output_channels)
                    n += 1

                    pbar.set_postfix(**{'m_IoU:': iou/n})
                    pbar.update(img.shape[0])

                    if idx == 0:
                        writer.add_images('images', img, epoch)
                        writer.add_images('masks/true', mask, epoch)
                        writer.add_images('masks/pred', mask_pred, epoch)

        writer.add_scalar('m_IoU/val', iou / n, epoch)

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:\n%(message)s')

    args = get_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    net = get_net(args.net, args.input_channels, args.output_channels).to(device=device)

    logging.info(f'\tNetwork: {args.net}\n'
                 f'\t{args.input_channels} input channels\n'
                 f'\t{args.output_channels} output channels (classes)\n')

    train_net(net, args, device)
