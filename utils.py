# 1. audio processing
import librosa
import soundfile as sf
from pydub import AudioSegment
import math
from pathlib import Path
import os

intermediate_results_dir = Path("data/intermediate")

# 1. audio processing
def get_audio_duration(filepath: str, ceil=True) -> float:
    """wrap up librosa.get_duration()
    """
    # exceptions handled by librosa

    orig_duration = librosa.get_duration(path=filepath) * 1000
    return orig_duration if not ceil else math.ceil(orig_duration)

def audio_segment(srcpath: Path, 
                startTime:int, 
                seglen:int, 
                src_duration: int,
                intermediate_dir: str,
                export=True,
                trgtpath=None) -> str:
    """wrap up AudioSegment.from_wav()[startTime:endTime]
       set default to export segment to file
    """
    endTime = min(startTime + seglen, src_duration)
    if trgtpath is None:
        trgtname = srcpath.stem + '_clip_' + str(startTime) + '_' + str(endTime) + '.wav'
        trgtpath = intermediate_dir / trgtname

    if Path(trgtpath).exists():
        return str(trgtpath)
    
    segment = AudioSegment.from_wav(srcpath)[startTime:endTime]
    segment.export(trgtpath, format="wav")

    return str(trgtpath)

# 2. text processing
stopchrs = [
    "、","。","〈","〉","《","》","︿","！","＃","＄","％","＆",\
    "（","）","＊","＋","，","：","；","＜","＞","？","＠","［",\
    "］","｛","｜","｝","～","￥","."
]

def strip_stopchrs(plain_text: str) -> str:
    res = ""
    for c in plain_text:
        if c not in stopchrs:
            res += c
    
    return res

def get_chn_char_dict(plain_text: str) -> dict:
    """Delete stop characters from Chinese plain text.
    """
    chn_char_list = [c for c in plain_text if c not in stopchrs]
    return dict(zip(range(len(chn_char_list)), chn_char_list))

def milisec_to_min(milisec: int) -> str:
    min, sec = milisec // 1000 // 60, (milisec // 1000) % 60 
    return f"{min}:{sec}"