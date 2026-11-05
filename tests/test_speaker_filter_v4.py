import numpy as np
import pandas as pd
from src.dataset.speaker_filter_v4 import prompt_like,masks
from src.dataset.contracts_v2 import screened_rows


def test_prompt_estimates_protect_narratives_and_quoted_questions():
    for text in ['how close are you to your family',"what's your dream job",'did you ever see','tell me about your work']:
        assert prompt_like(text)
    for text in ['what I wanted to do was work with you','I asked her how close are you to your family',
                 'set a good example because you are the first teacher','anything that pays','yes','they lied to me',
                 'How did you know I was there?']:
        assert not prompt_like(text)


def test_context_support_uses_chronology_and_retains_uncertain_answers():
    f=pd.DataFrame({'Start_Time':[2.,0.,4.,50.],'End_Time':[3.,1.,5.,51.],
                    'Text':['I work in construction',"what's your dream job",'did you ever see','yes']})
    rows=screened_rows(f,60);keep=masks(f,rows)
    np.testing.assert_array_equal(keep['prompt_filter'],[True,False,False,True])
    np.testing.assert_array_equal(keep['context_filter'],[True,False,True,True])
