import argparse


def get_argparser():
    parser = argparse.ArgumentParser(description="Traing a grading network")

    parser.add_argument("--train_root", type=str, default='/home3/ljc/datasets/GAMMA_dataset/multi-modality_images',
                        help="path to Dataset")
    parser.add_argument("--label_file", type=str,
                        default='/home3/ljc/datasets/GAMMA_dataset/glaucoma_grading_training_GT.xlsx',
                        help="path to label file")  # 'E:/dataset/GAMMA_training data/training_data/glaucoma_grading_training_GT.xlsx'

    parser.add_argument("--oct_img_size", type=list, default=[512, 512],
                        help="the size of oct_img")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--bs", type=int, default=4,
                        help="batch_size = 4")
    parser.add_argument("--image_size", type=int, default=512,
                        help="the size of img_train_transforms")
    parser.add_argument("--iters", type=int, default=1000,
                        help="iters of train phase")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="val_ratio: 80 / 20")
    parser.add_argument("--test_root", type=str, default="",
                        help="path to test Dataset")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="num_workers")
    parser.add_argument("--optimizer_type", type=str, default="adam",
                        help="optimizer_type")
    parser.add_argument("--loss_type", type=str, default="ce", choices=['ce', 'fl'],
                        help="loss_type")
    parser.add_argument("--model_mode", type=str, default="18", choices=['18', '34', 'x50', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
                        help="model_mode")
    parser.add_argument("--test_mode", type=str, default="test", choices=['val', 'test'],
                        help="val or test mode")
    parser.add_argument("--sub1dataset_mode", type=str, default="crop", choices=['orig', 'crop', 'crop_cup', 'small'],
                        help="")
    parser.add_argument("--transforms_mode", type=str, default="centerv2", choices=['orig', 'center', 'centerv2'],
                        help="val or test transforms_mode")
    parser.add_argument("--gaussian", type=str, default="", choices=['', '_gaussian'],
                        help="fundus img")
    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=3, type=float)
    parser.add_argument("--eval_kappa", type=str, default="9999",
                        help="eval_kappa")

    return parser
