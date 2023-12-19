import json
import tqdm

def format_options(options):
  sorted_keys = sorted(options.keys(), key=lambda x: int(x))
  formatted_options = [f"{chr(65 + int(key))}){options[key]}" for key in sorted_keys]
  options_string = "\n".join(formatted_options)
  return formatted_options

"""
Concatenate options and questions
"""
def format_data(data, start, end, few_shots = " ", part = None, question = 'question', options_col = 'options'):
    new_data = []
    if end == -1:
        if part!=None:
            end = len(data[part])
        else:
            end = len(data)

    for i in range(start, end):
        new_row = {}
        # if else depending on weather data has a train-test split or not
        if part!=None:
            q = data[part][question][i]
            options = eval(data[part][options_col][i])
        else:
            q = data[question][i]
            options = eval(data[options_col][i])
        # start making the new datapoint
        new_row['question'] = q
        correct_answer = data['correct_answer'][i]
        new_row['correct_answer'] = correct_answer
        correct_idx = data['correct_index'][i]
        new_row["correct_idx"] = correct_idx
        # add the options
        if "correct answer" in options:
            del options['correct answer']
        _options = format_options(options)
        new_row["options"] = _options
        # create the prompt
        prompt = q + "\n".join(_options)
        new_row["prompt"] = prompt
        new_row["subject_name"] = data["subject_name"][i]

        new_data.append(new_row.copy())
    return new_data


"""
write data
"""
def write_data(data, path):
    with open(path, "w") as file:
        if path.split(".")[-1] == "json":
            json.dump(data, file)
        else:
            file.write(data)