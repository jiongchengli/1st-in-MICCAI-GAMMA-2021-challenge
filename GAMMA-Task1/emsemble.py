def init_scoredict_1():
    dict = {}
    with open('./final_results/confidence_test_orig_centerv2_x50_8993.csv') as f:
        for i, line in enumerate(f):
            if i > 0:
                a, b, c, d = line.rstrip().split(',')
                dict[a] = [0, 1, 0]
    print(dict)
    return dict


def scoredict2cls(score_dict):
    for i in score_dict:
        score_dict[i] = score_dict[i].index(max(score_dict[i]))
    return score_dict

def countdict(dict):
    num0 = 0
    num1 = 0
    num2 = 0
    for i in dict:
        if dict[i] == 0:
            num0 += 1
        elif dict[i] == 1:
            num1 += 1
        else:
            num2 += 1
    print(num0, num1, num2)

def dict2result(clsdict):
    idxlist = []
    nonlist = []
    earlylist = []
    advancedlist = []

    for i in clsdict:
        idxlist.append(i)
        if clsdict[i] == 0:
            nonlist.append(1)
            earlylist.append(0)
            advancedlist.append(0)
        elif clsdict[i] == 1:
            nonlist.append(0)
            earlylist.append(1)
            advancedlist.append(0)
        else:
            nonlist.append(0)
            earlylist.append(0)
            advancedlist.append(1)

    with open('./Classification_Results.csv', "w+") as f:
        f.write("{},{},{},{}\n".format('data', 'non', 'early', 'mid_advanced'))  # 一般就是表头，就是对应的列名
        for i in range(len(idxlist)):
            f.write("{},{},{},{}\n".format(idxlist[i], nonlist[i], earlylist[i], advancedlist[i]))  # 每行需要写入的内容


def emsemble_final():
    dict_score_v1 = init_scoredict_1()
    print(dict_score_v1)
    csvfile = './final_results/confidence_test_orig_centerv2_x50_8993.csv'
    with open(csvfile) as f:
        for i, line in enumerate(f):
            if i > 0:
                # print(line.rstrip().split(','))
                a, b, c, d = line.rstrip().split(',')
                if float(d) >= 0.6:
                    print(a)
                    dict_score_v1[a] = [0, 0, 1]
    print(dict_score_v1)

    csvfile = './final_results/pred_test_crop_centerv2_34_8438.csv'
    with open(csvfile) as f:
        for i, line in enumerate(f):
            if i > 0:
                # print(line.rstrip().split(','))
                a, b, c, d = line.rstrip().split(',')
                if b == '1':
                    dict_score_v1[a] = [1, 0, 0]


    csvfile = './final_results/confidence_test_orig_centerv2_x50_8625.csv'
    with open(csvfile) as f:
        for i, line in enumerate(f):
            if i > 0:
                # print(line.rstrip().split(','))
                a, b, c, d = line.rstrip().split(',')
                if float(c) >= 0.90:
                    print(a)
                    dict_score_v1[a] = [0, 1, 0]

    print(dict_score_v1)
    dict_score_v1 = scoredict2cls(dict_score_v1)
    print(dict_score_v1)
    countdict(dict_score_v1)
    dict2result(dict_score_v1)

emsemble_final()