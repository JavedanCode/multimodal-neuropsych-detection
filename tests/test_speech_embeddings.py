import io
import wave

import numpy as np
import pytest

from src.dataset.speech_embeddings import window_starts,decode_audio,pool_windows,WINDOW


def test_window_sampling_is_bounded_even_and_nonoverlapping():
    for seconds in [8,12,16,60,512,1000]:
        samples=seconds*16000;starts=window_starts(samples)
        assert len(starts)==min(64,seconds//8)
        assert starts[0]==0 and starts[-1]+WINDOW<=samples
        if len(starts)>1:
            assert starts[-1]+WINDOW==samples
            assert (np.diff(starts)>=WINDOW).all()
            assert np.ptp(np.diff(starts))<=1
    with pytest.raises(ValueError,match='shorter'):window_starts(WINDOW-1)


def test_stereo_downmix_and_resampling_are_local_and_exact_length():
    values=np.tile(np.array([[1000,-1000]],dtype='<i2'),(48000,1))
    output=io.BytesIO()
    with wave.open(output,'wb') as wav:
        wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(48000);wav.writeframes(values.tobytes())
    audio,meta=decode_audio(output.getvalue())
    assert len(audio)==16000 and meta['source_rate']==48000 and meta['channels']==2
    assert np.array_equal(audio,np.zeros(16000))


def test_speech_pooling_preserves_between_window_variation():
    x=np.stack([np.zeros(768),np.ones(768)*4])
    result=pool_windows(x)
    assert len(result)==1536
    assert result['audio__dense_wav2vec_0__mean']==2
    assert result['audio__dense_wav2vec_767__std']==2
    with pytest.raises(ValueError,match='Invalid'):pool_windows(np.ones((2,767)))
    x[0,0]=np.nan
    with pytest.raises(ValueError,match='Invalid'):pool_windows(x)
