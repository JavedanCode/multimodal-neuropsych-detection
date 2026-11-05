from src.experiments.encoder_blend import combinations


def test_complementary_scores_can_outperform_either_endpoint():
    scores={'mpnet':[{'scores':[2,-.2,-2,1]}],'minilm':[{'scores':[-.2,2,1,-2]}]}
    records=combinations(scores,[1,1,0,0])
    best=max(records,key=lambda r:r['inner_f1'])
    assert best['weight']==.5 and best['inner_f1']==1
    assert records[0]['inner_f1']<1 and records[-1]['inner_f1']<1


def test_ties_prefer_original_encoder():
    scores={'mpnet':[{'scores':[1,-1]}],'minilm':[{'scores':[2,-2]}]}
    assert max(combinations(scores,[1,0]),key=lambda r:r['inner_f1'])['weight']==1
