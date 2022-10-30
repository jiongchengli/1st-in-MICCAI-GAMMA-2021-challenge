import torch
from lib.model.dual_resnet import Dual_network18, Dual_network34, Dual_networkx50
import pandas as pd
import torchvision.transforms as trans
from lib.datasets.sub1 import GAMMA_sub1_dataset, GAMMA_sub1_dataset_512, GAMMA_sub1_dataset_224
import torch.nn.functional as F
import pdb
from lib.model.utils.config import get_argparser
from tools.analysis_tools.pred_analysis import analyse_num



def main():
    args = get_argparser().parse_args()

    # best_model_path = "./models/dual_resnet18_512_centerv2/best_model100_0.8649.pth"
    # best_model_path = "./models/dual_resnet34_orig_centerv2_gaussian/best_model100_0.8276.pth"
    # best_model_path = './models/dual_resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + 'fl3.0/best_modelacc12_0.7273.pth'


    if args.model_mode == 'b0' or args.model_mode == 'b1' or args.model_mode == 'b2' or args.model_mode == 'b3' or args.model_mode == 'b4' or args.model_mode == 'b5' or args.model_mode == 'b6' or args.model_mode == 'b7':
        best_model_path = './final_models/dual_efficientnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + '/best_model100_0.' + args.eval_kappa + '.pth'
    else:
        best_model_path = './final_models/dual_resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + '/best_model100_0.' + args.eval_kappa + '.pth'

    image_size = args.image_size  # 256

    if args.test_mode == 'val':
        testset_root = '/home3/ljc/datasets/GAMMA_dataset/multi-modality_images'
    elif args.test_mode == 'test':
        testset_root = '/home3/ljc/datasets/GAMMA_dataset/val/val_data/multi-modality_images'

    model = torch.load(best_model_path)
    model = model.cuda()

    model.eval()

    if args.transforms_mode == 'centerv2':
        img_test_transforms = trans.Compose([
            # trans.CropCenterSquare(),
            trans.CenterCrop((2000, 2000)),  ###v2c
            trans.Resize((image_size, image_size)),
            trans.ToTensor(),
        ])

    oct_test_transforms = trans.Compose([
        # trans.CenterCrop(512),
        trans.ToTensor(),
    ])

    if args.sub1dataset_mode == 'orig':
        test_dataset = GAMMA_sub1_dataset(dataset_root=testset_root,
                                          img_transforms=img_test_transforms,
                                          oct_transforms=oct_test_transforms,
                                          mode='test')
    elif args.sub1dataset_mode == 'crop':
        test_dataset = GAMMA_sub1_dataset_512(dataset_root=testset_root,
                                              img_transforms=img_test_transforms,
                                              oct_transforms=oct_test_transforms,
                                              mode='test')
    elif args.sub1dataset_mode == 'small':
        test_dataset = GAMMA_sub1_dataset_224(dataset_root=testset_root,
                                              img_transforms=img_test_transforms,
                                              oct_transforms=oct_test_transforms,
                                              mode='test')

    val = ['0080', '0072', '0023', '0089', '0017', '0075', '0097', '0064', '0021', '0049', '0065', '0047', '0036',
           '0014', '0076', '0055', '0083', '0092', '0063', '0082']

    cache = []

    idxlist = []
    prob1list = []
    prob2list = []
    prob3list = []

    for fundus_img, oct_img, idx in test_dataset:
        # print(idx)
        # print(type(idx))
        # print(fundus_img.size())
        # print(oct_img.size())

        fundus_img = fundus_img.unsqueeze(0).cuda()
        oct_img = oct_img.unsqueeze(0).cuda()

        if args.test_mode == 'val':
            if idx in val:  # 只取
                print(idx)
                logits = model(fundus_img, oct_img)
                # print(logits)
                cls_prob = F.softmax(
                    logits)  ####tensor([[0.0044, 0.0025, 0.9931]], device='cuda:0', grad_fn=<SoftmaxBackward>)
                # pdb.set_trace()
                print(cls_prob)
                prob1, prob2, prob3 = float(cls_prob[0][0].detach().cpu().numpy()), float(
                    cls_prob[0][1].detach().cpu().numpy()), float(cls_prob[0][2].detach().cpu().numpy())
                idxlist.append(idx)
                prob1list.append(prob1)
                prob2list.append(prob2)
                prob3list.append(prob3)

                # cache.append([idx, logits.numpy().argmax(1)])
                cache.append([idx, logits.detach().cpu().numpy().argmax(1)])
        else:
            print(idx)
            logits = model(fundus_img, oct_img)
            # print(logits)
            cls_prob = F.softmax(
                logits)  ####tensor([[0.0044, 0.0025, 0.9931]], device='cuda:0', grad_fn=<SoftmaxBackward>)
            # pdb.set_trace()
            print(cls_prob)
            prob1, prob2, prob3 = float(cls_prob[0][0].detach().cpu().numpy()), float(
                cls_prob[0][1].detach().cpu().numpy()), float(cls_prob[0][2].detach().cpu().numpy())
            idxlist.append(idx)
            prob1list.append(prob1)
            prob2list.append(prob2)
            prob3list.append(prob3)

            # cache.append([idx, logits.numpy().argmax(1)])
            cache.append([idx, logits.detach().cpu().numpy().argmax(1)])

    suffix_name = args.test_mode + "_" + args.sub1dataset_mode + "_" + args.transforms_mode + "_" + args.model_mode + "_" + best_model_path[
                                                                                                                            -8:-4] + ".csv"

    confidence_file = './final_results/confidence_' + suffix_name
    result_file = "./final_results/pred_" + suffix_name

    with open(confidence_file, "w+") as f:
        f.write("{},{},{},{}\n".format('data', 'non', 'early', 'mid_advanced'))  # 一般就是表头，就是对应的列名
        for i in range(len(idxlist)):
            f.write("{},{},{},{}\n".format(idxlist[i], prob1list[i], prob2list[i], prob3list[i]))  # 每行需要写入的内容

    submission_result = pd.DataFrame(cache, columns=['data', 'dense_pred'])

    submission_result['non'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 0))
    submission_result['early'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 1))
    submission_result['mid_advanced'] = submission_result['dense_pred'].apply(lambda x: int(x[0] == 2))

    submission_result[['data', 'non', 'early', 'mid_advanced']].to_csv(
        result_file,
        index=False)

    analyse_num(confidence_file)


if __name__ == '__main__':
    main()
