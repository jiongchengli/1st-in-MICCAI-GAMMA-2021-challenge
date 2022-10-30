import pandas as pd

def analyse_num(label_file):
    label = {str(int(row['data'])).rjust(4, '0'): row[1:].values.argmax()
             for _, row in pd.read_csv(label_file).iterrows()}
    # print(label)
    num0 = 0
    num1 = 0
    num2 = 0
    for i in label:
        # print(label[i])
        # print(type(int(label[i])))
        if int(label[i]) == 0:
            num0 += 1
        elif int(label[i]) == 1:
            num1 += 1
        else:
            num2 += 1
    print(num0, num1, num2)
    return label

