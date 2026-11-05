import pandas as pd
import pytest
from src.dataset.transcript_context_v4 import context


def test_context_uses_chronology_not_source_row_order():
    frame=pd.DataFrame({'Start_Time':[4.,0.,2.],'End_Time':[5.,1.,3.],
                        'Text':['later','earlier','target']})
    rows=context(frame,2.,3.)
    assert [r['text'] for r in rows]==['earlier','target','later']
    assert rows[1]['target'] and rows[1]['source_row']==2
    assert sum(r['target'] for r in rows)==1


def test_context_rejects_truncated_or_ambiguous_target():
    frame=pd.DataFrame({'Start_Time':[1.],'End_Time':[15.],'Text':['full utterance']})
    with pytest.raises(ValueError):context(frame,1.,9.)
    with pytest.raises(ValueError):context(pd.concat([frame,frame],ignore_index=True),1.,15.)
