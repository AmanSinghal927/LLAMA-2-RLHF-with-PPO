import os
from tqdm import tqdm
import datasets
import time
from utils.data import format_data, write_data
from utils.api_call import togther_api
from datasets import load_dataset


if __name__ == '__main__':
    path = r"C:\Users\J C SINGLA\PycharmProjects\olab\mmlu_test"

    # medhalt_fct = load_dataset('MedHALT/Med-HALT', 'reasoning_FCT')
    # medhalt_fct = medhalt_fct['train'].train_test_split(test_size=0.1)
    # medhalt_fct['train'].save_to_disk(os.path.join(path, "train.json"))
    # medhalt_fct['test'].save_to_disk(os.path.join(path, "test.json"))
    # medhalt_fct_train = datasets.load_from_disk(os.path.join(path, "train.json"))
    mmlu_test = datasets.load_from_disk(path)

    if os.path.exists(os.path.join(path, "prev_stop_1.txt")):
        with open(os.path.join(path, "prev_stop_1.txt"), "r") as file:
            prev_stop = file.read()
            prev_stop = int(prev_stop)
    else:
        prev_stop = 0

    new_data = []
    pbar = tqdm(range(prev_stop, len(mmlu_test)))

    for i in pbar:
        try:
            few_shots = ["\n\n####user:\n For what age group is the turtle technique indicated? A) Young children (Early Childhood Education and first courses of Primary Education).\nB) Older children (last courses of Primary Education).\nC) Preadolescents (first courses of Secondary Education).\nD) Adolescents (last years of Secondary Education and Bachillerato).\n####assistant:\n The correct answer is Older children (last courses of Primary Education).\n####user:\n Why is that?\n####assistant:\nThe Turtle Technique is primarily indicated for older children, specifically those in the last courses of Primary Education. This approach is often used to help children in this age group develop better self-control and emotional regulation skills."]
            ex = format_data(mmlu_test, i, (i+1), few_shots = None, part = None)
            new_data.append(ex[0])
            # print (new_data)

            output0 = togther_api(ex[0]["prompt"], option = ex[0]["options"][0], verbose = False)
            output1 = togther_api(ex[0]["prompt"], option = ex[0]["options"][1], verbose = False)
            output2 = togther_api(ex[0]["prompt"], option = ex[0]["options"][2], verbose = False)
            output3 = togther_api(ex[0]["prompt"], option = ex[0]["options"][3], verbose = False)
            new_data[i-prev_stop]["output0"] = output0
            new_data[i-prev_stop]["output1"] = output1
            new_data[i-prev_stop]["output2"] = output2
            new_data[i-prev_stop]["output3"] = output3


            if i%10 == 0 and i > 0:
                write_data(new_data, os.path.join(path,"generations", "zero_shot", "_" + str(prev_stop) + ".json"))
                print ("Writing")
                new_data = []
                prev_stop = i+1
                write_data(str(prev_stop), r"C:\Users\J C SINGLA\PycharmProjects\olab\medhalt\prev_stop_1.txt")
        except:
            print ("\nSleeping")
            time.sleep(3)












# See PyCharm help at https://www.jetbrains.com/help/pycharm/
