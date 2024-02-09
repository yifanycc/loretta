
#         Premise:  It was a complex language. Not written down but handed down. One might say it was peeled down.
# Hypothesis: the language was peeled down	Output:0
#
# Premise: It is part of their religion, a religion I do not scoff at as it holds many elements which match our own even
# though it lacks the truth of ours. At one of their great festivals they have the ritual of driving out the devils
#  from their bodies. Fist the drummers come on - I may say that no women are allowed to take part in this
#   ritual and the ladies here will perhaps agree with me that they are fortunate in that omission.
# Hypothesis: no women are allowed to take part in this ritual	Output:0
#
# Premise: He's weird enough to have undressed me without thinking, according to some mad notion of the ``proper'' thing to do. Perhaps he thought I couldn't lie in bed with my clothes on.
# Hypothesis: she couldn't lie in bed with her clothes on	Output:1
#
# Premise: It is all very well, in these changing times, to adapt one's work to take in duties not traditionally within one's realm. But bantering is of another dimension altogether. For one thing how would one know for sure that at any given moment a response of the bantering sort is truly what is expected?
# Hypothesis: at any given moment a response of the bantering sort is truly what is expected	Output:2
#
# Premise: If there are spirits at work at the time, they come only from yourself, not from the fume of the incense. Why should spirits aid living beings? What arrogance is it that drives people to believe they can have power over them?
# Hypothesis: people can have power over spirits	Output:0

def build_message(task_name, eval_sample):
    if task_name == 'CB':
        premise = eval_sample.data['premise']
        hypothesis = eval_sample.data['hypothesis']
        system_message = f"""
         This is a text entailment task. You will be given a premise and a hypothesis.
         Your job is to determine if the hypothesis is true (entailment), false (contradiction),
          or if there isn’t enough information (neutral). Return an integer 0 to represent entailment, 1 for contradiction
          and 2 for neutral. 
          
        only show a number
         """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"Premise: {premise}\n\nhypothesis: {hypothesis}"}
        ]
        label = eval_sample.correct_candidate
    elif task_name == 'BoolQ':
        passage = eval_sample.data['passage']
        question = eval_sample.data['question']
        system_message = f"""
                 You will be presented with a passage and a yes/no question based on that passage. 
                 Read the passage and answer the question with either 'Yes' or 'No'.
                 
                 only return 'yes' or 'no'
                 """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"passage: {passage} \n\nquestion: {question}"}
        ]
        label = eval_sample.correct_candidate
    elif task_name == 'WSC':
        span1_text = eval_sample.data['span1_text']
        span2_text = eval_sample.data['span2_text']
        text = eval_sample.data['text']
        system_message = f"""
                 Giving a sentence, does the pronoun <span2> refer to <span1>? Yes or No?
                 
                 return 0 if you choose No and 1 if you choose yes
                 
                 return only the number
                 """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"sentence: {text}\n\n<span1>: {span1_text}\n\n<span2>: {span2_text}"}
        ]
        label = eval_sample.correct_candidate
    elif task_name == 'Copa':
        premise = eval_sample.data['premise']
        alternative_1 = eval_sample.data['choice1']
        alternative_2 = eval_sample.data['choice2']
        system_message = f"""
                This task involves choosing the most plausible alternative. You will be given a premise and two alternatives. 
                Your job is to select the alternative that is more plausible based on the premise. 
                
                only return the text in the alternative you choose
                 """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"premise: {premise}\n\nalternative 1: {alternative_1} alternative 2: {alternative_2}"}
        ]
        label = eval_sample.correct_candidate
    elif task_name == 'ReCoRD':
        passage = eval_sample.data['passage']
        question = eval_sample.data['query']
        entities = eval_sample.data['entities']
        system_message = f"""
                This is a reading comprehension task with a focus on commonsense reasoning. You will be given a passage 
                and a question with a blank to fill in (the blank is represent as @placeholder). Your job is to fill in the blank by choosing one word in the entities given. 
                 
                 return only the word you choose
                 """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"passage: {passage}\n\nquestion: {question} entities {entities}"}
        ]
        label = eval_sample.correct_candidate
    elif task_name == 'SQuAD':
        title= eval_sample.data['title']
        context = eval_sample.data['context']
        question = eval_sample.data['question']
        system_message = f"""
                 his task involves answering questions based on a given passage and title. For each passage, 
                 you will be presented with a question, and your job is to find and return the answer from the passage.
                  
                  return only the answer of the question
                 """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"Title: {title} passage: {context}\n\nquestion: {question}"}
        ]
        label = eval_sample.correct_candidate
    elif task_name == 'DROP':
        passage = eval_sample.data['context']
        question = eval_sample.data['question']
        system_message = f"""
                 This is a task that requires numerical reasoning over text. You will be given a passage and a question 
                 that requires numerical understanding (such as counting, addition, subtraction) to answer. 
                 Find the answer from the passage and provide it. 
                 
                 return only the answer of the question
                 """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"passage: {passage}\n\nquestion: {question}"}
            ]
        label = eval_sample.correct_candidate
    return messages, label
