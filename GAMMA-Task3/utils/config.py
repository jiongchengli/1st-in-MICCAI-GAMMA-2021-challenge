import argparse


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--dataset", type=str, default='refuge')
    parser.add_argument("--num_classes", type=int, default=3)

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3_resnet34',
                        choices=['deeplabv3_resnet34', 'deeplabv3_efficientnet_b2'])
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default='ce',
                        choices=['ce', 'dc_ce'])
    parser.add_argument("--lr", type=float, default=1e-3)

    # Other Options
    parser.add_argument("--valid", action='store_true', default=False)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--ckpt", type=str, default=None)

    return parser
