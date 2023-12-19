import json
import together
together.api_key = "32a9ddb4b82781cbf7870cc604c6a0e186c30259741d148a8198a2fe98cedab3"


def togther_api(prompt, option = None, verbose = False, model = "togethercomputer/llama-2-13b-chat"):
    if option == None:
        prompt = "<human>:" + prompt
    else:
        prompt = "\n\n####user:\n" + prompt + "\n####assistant:\n The correct answer is " + option + "\n####user:\n Why is that?" + "\n####assistant:\n"

    if verbose:
        print (prompt)

    output = together.Complete.create(
        prompt=prompt,
        model=model,
        max_tokens=1024,
        temperature=0.7,
        top_k=50,
        top_p=0.7,
        repetition_penalty=1,
        stop=['<human>', '\n\n']
    )
    return (output['output']['choices'][0]['text'])


def togther_api_few_shots(prompt, option = None, few_shots = None, verbose = False, model = "togethercomputer/llama-2-13b-chat"): #"togethercomputer/llama-2-70b-chat"): #
    if option == None:
        prompt = "\n####user:" + prompt
    else:
        prompt = few_shots + "\n\n####user:\n" + prompt + "\n####assistant:\n The correct answer is " + option + "\n####user:\n Why is that?" + "\n####assistant:\n"

    if verbose:
        print (prompt)

    output = together.Complete.create(
        prompt=prompt,
        model=model,
        max_tokens=1024,
        temperature=0.7,
        top_k=50,
        top_p=0.7,
        repetition_penalty=1,
        stop=['<human>', '\n\n']
    )
    return (output['output']['choices'][0]['text'])
