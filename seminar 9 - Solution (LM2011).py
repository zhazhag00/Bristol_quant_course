
import pandas as pd
import numpy as np
import pysentiment2 as ps
import ast

# load data
PATH_2_data = ".\\"
wsj = pd.read_csv(PATH_2_data + "wsj.csv")

# convert the string literal of lists into real lists
wsj['sentences'] = wsj['sentences'].apply(lambda x: ast.literal_eval(x) )

print(wsj['text'].values[0])

#############
# News Tone # - LM Dict
#############
# i.e., NegTone, NewsTone, UncerTone

# load LM dictionary method
lm = ps.LM()

def LM_sent(text):
    tokens = lm.tokenize(text)
    score = lm.get_score(tokens)
    print(score)
    return score['Negative'], score['Polarity'],len(tokens)

# calculate the different type of article tone
wsj[['neg_word','news_tone','#_of_words']] = wsj['text'].apply(lambda x: pd.Series(LM_sent(x)))

# the proportion of negative words
wsj['neg_tone'] = wsj['neg_word']/wsj['#_of_words']

wsj.to_csv(PATH_2_data + "LM_result.csv",index=False)






