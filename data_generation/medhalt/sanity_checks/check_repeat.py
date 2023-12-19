import json
import glob
import os

def unq_questions(path = r"C:\Users\J C SINGLA\PycharmProjects\olab\medhalt", fn = "generations"):
    """
    function to check if answers have been generated for unique questions
    """
    data = []
    for i in glob.glob(os.path.join(path, fn,"*.json")):
        with open(i, "r") as file:
            temp = json.load(file)
        data.extend(temp)
    questions = [i["question"] for i in data]
    return (len(data) == len(list(set(questions))))


# if __name__ == '__main__':
#     print (unq_questions())





