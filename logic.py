#coding:utf-8

from utils import *
import json

from funasr import AutoModel
import numpy as np
from openai import OpenAI

import re

# Configurations
audio_dir = Path("data/raw")
# transcript_dir = Path("data/transcription")
model_dir = Path("E:/SenseVoice_cache/iic")
model_filename = {
    "asr_model": "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "punc_model": "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "vad_model": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
}
model_revision = {
    "asr_model": "v2.0.4",
    "punc_model": "v2.0.4",
    "vad_model": "v2.0.4",
}


def build_text2audioToken_lookup_table(
        text: str, 
        timestamp: list[list]) -> tuple:
    """build one-on-one lookup table for full text to audioTokens(timestamp)
    """
    chn_chr_dict = get_chn_char_dict(text)
    audioToken_starts = (np.array(timestamp, dtype='int')[:,0]).tolist()
    
    '''
    assert len(chn_chr_dict) >= len(audioToken_starts), \
            "audioToken outnumber char"
    '''
    
    # NOTE: under tests, finer-grained alignment -> 2000ms
    # calculate speaking rate of the audio, estimate English word/sec

    return dict(zip(chn_chr_dict.keys(), audioToken_starts)), len(chn_chr_dict), len(audioToken_starts)

def helper_full_query_matching_retrieval(context, query: str,
                                    recur: bool,
                                    finer_seg_duration=10000
                                ) -> dict:

    fulltext, timestamps = context.get_full_stripped_transcript(), context.seg_lookup_tbl
    seg_chr_lens = [len(timestamps[k]) for k in timestamps]
    # print(fulltext)

    matched = re.finditer(query, fulltext)

    res = {query: []}
    for m in matched:
        query_order = m.span()[0] + 1
        for seg_i, seg_chr_len in enumerate(seg_chr_lens):
            if query_order - seg_chr_len <= 0:
                break
            query_order -= seg_chr_len
        
        # located segment is seg_i
        if recur or len(timestamps[str(seg_i)].keys()) == len(timestamps[str(seg_i)].values()):
            # CASE: segment is not misaligned
            crnt_timestamp = seg_i * context.seg_duration + timestamps[str(seg_i)][str(query_order)]
            res[query].append(milisec_to_min(crnt_timestamp))

        else:
            # CASE: segment misaligned, perform finer-grained alignment
            trgtname = context.srcpath.stem / ('_clip_' + str(seg_i*context.seg_duration))
            seg_path = context.intermediate_dir / trgtname
            assert seg_path.exists(), f"Cannot find the segment {seg_path}"
            
            segAudioTrans = AudioFileTranscript(seg_path, finer_seg_duration)
            helper_full_query_matching_retrieval(context=segAudioTrans,
                                                 query=query,
                                                 recur=True)

    
    return res

class AudioFileTranscript():
    """Class for transcripted full audio file
    attributes:
        srcpath
        duration
        transcript: full transcrcipt of audio file
        lookup_table: each (audioToken <-> chn_char)
    """
    
    def __init__(self, srcpath:str, seg_duration = 60000) -> None:
        self.srcpath = Path(srcpath)
        self.duration = get_audio_duration(srcpath) # in ms

        self.seg_duration = seg_duration

        self.intermediate_dir = intermediate_results_dir / self.srcpath.stem
        self.intermediate_dir.mkdir(parents=False, exist_ok=True)

        self.model = None
        if Path(self.intermediate_dir / 'seg_transcripts.json').exists():
            with open(self.intermediate_dir / 'seg_transcripts.json') as fp:
                self.trans_segments_info = json.load(fp)

        else:
            self.trans_segments_info = self.transcribe_segments()
            self.export_transcipt_info()

        self.full_stripped_text = self.get_full_stripped_transcript()

        if not Path(self.intermediate_dir / 'seg_lookup_tbl.json').exists():
            self.build_char_audioToken_lookup_tbl(export=True)
        with open(self.intermediate_dir / 'seg_lookup_tbl.json') as fp:
            self.seg_lookup_tbl = json.load(fp)
            
    def transcribe_audio(self, filepath: str) -> dict:
        """transcribe audio wav file with model. 
        return transcription info.
        """
        if self.model is None:
            asr_model_path, vad_model_path, punc_model_path, = \
                model_dir / model_filename["asr_model"], \
                model_dir / model_filename["vad_model"], \
                model_dir / model_filename["punc_model"]

            self.model = AutoModel(model=str(asr_model_path), model_revision=model_revision["asr_model"],
                            vad_model=str(vad_model_path), vad_model_revision=model_revision["vad_model"],
                            punc_model=str(punc_model_path), punc_model_revision=model_revision["punc_model"],
                            disable_update=True,
                        )
        
        res = self.model.generate(input=filepath,
                            batch_size_s=300,   # duration of audio to transcribe, in sec.
                        )
        return res 

    def transcribe_segments(self) -> dict:
        """segment audio file to 1min clips and transcribe each clip,
           return transiption_dict:
           {
                1: {
                        segment_path: ,
                        start_in_src: ,
                        end_in_src: ,
                        duration: ,
                        text:, # with punctuation
                        timestamp: ,

                }
            }
        """
        _segintv = self.seg_duration
        
        seg_transcripts = dict()
        
        seg_intv = self.duration // _segintv
        for i, startTime in enumerate(range(seg_intv + 1)):
            seg_transcripts[i] = dict()

            startTime *= _segintv
            seg_path = audio_segment(self.srcpath, 
                            startTime=startTime, 
                            seglen=_segintv,
                            intermediate_dir=self.intermediate_dir,
                            src_duration=self.duration
                        )
            
            seg_transcripts[i]["start_in_src"] = startTime
            seg_transcripts[i]["segment_path"] = seg_path

            trans_res = self.transcribe_audio(seg_path)[0]

            seg_transcripts[i]['text'] = trans_res['text']
            seg_transcripts[i]['timestamp'] = trans_res['timestamp']

        return seg_transcripts

    def get_full_stripped_transcript(self) -> str:
        """Full transcription of audio file, stopchars stripped.
        """
        full_transcript = ""
        for _, seginfo in self.trans_segments_info.items():
            full_transcript += strip_stopchrs(seginfo['text'])
        
        return full_transcript
    
    def build_char_audioToken_lookup_tbl(self, export=False, export_dir=None) -> None:
        """build char-to-audioToken lookup table,
           if export=True, dump a json file of the lookup table
        """
        self.seg_lookup_tbl = dict()

        for i, seg_info in self.trans_segments_info.items():
            self.seg_lookup_tbl[str(i)], _, _ = build_text2audioToken_lookup_table(seg_info['text'], seg_info['timestamp'])

        if export:
            if export_dir is None:
                export_dir = self.intermediate_dir / 'seg_lookup_tbl.json'
            with open(export_dir, mode="a") as f:
                json.dump(self.seg_lookup_tbl, f, ensure_ascii=False)    

    def export_transcipt_info(self, export_dir=None) -> None:
        """dump segmented transcription info into .json file.
        """
        if export_dir is None:
            export_dir = self.intermediate_dir / 'seg_transcripts.json'
        with open(export_dir, mode="a") as f:
            json.dump(self.trans_segments_info, f, ensure_ascii=False)

class QueryRetrieval():
    def __init__(self, context:AudioFileTranscript) -> None:
        self.context = context
        # mode = {"Full Query Matching": 0, "Vague Search": 1}
        # self.mode = mode[modedigit]
        self.queries = {'Full Query Matching':[], 'Vague Search':[],}
    
    def full_query_matching_retrieval(self, query: str, 
                                      from_vague_retriv: bool,
                                      finer_seg_duration=10000
                                    ) -> dict:
        if not from_vague_retriv:
            self.queries['Full Query Matching'].append(query)

        return helper_full_query_matching_retrieval(context=self.context,
                                                    query=query,
                                                    recur=False,
                                                    finer_seg_duration=finer_seg_duration
                                                )

    def vague_query_retrieval(self, usrQuery: str) -> int:
        """vague retrieve audio timestamp with user query
        """
        self.queries['Vague Search'].append(usrQuery)

        client = OpenAI(api_key="your-own-api", 
                        base_url="https://api.deepseek.com")

        system_prompt = """
        用户将提供一篇文本与一个问题。判断文本与问题的相关性，以0.00-1.00浮点数表示。
        IF(相关性 >= 0.2)
            从文本中找到与问题相关的部分，请尽可能多切分部分，各部分在文本内距离尽量较远，间隔尽量大于100字，返回每一部分的起始短句，长度为30-50字。返回格式为json对象，外部由""扩起，表示为字符串。
    
        ELSE
            返回json对象内sentence域为空字符串。

        输入示例1: 
        文本：三观正直一把尺扎扎实实讲历史我是李正欢迎收看正直讲史咱们这期聊聊今年的现象级历史剧觉醒年代那毫无疑问这部剧是近几年难得的优秀历史剧我自己在看的时候就想着一定要出一期视频来和大家聊一聊那现在b站中关于觉醒年代的视频非常多了啊有剧情解说也有那种高燃的混解不过身为一名历史老师我还是想从历史专业的角度和大家聊一聊觉醒年代背后的历史脉络首先觉醒年代的时间跨度首从一九一五年的新文化运动到一九二一年中共建党一共七年时间按理说这段历史其实是一个冷门的阶段因为我们之前往往更关注的都是党诞生后的故事很少关注党诞生前发生了什么事就比如现在如果我们让一个人形容一下新文化运动那他会说什么呢我猜可能比较多的还是高中课本上那些知识点像北大新青年然后各种代表人物再有就是经典的三提倡三反对提倡民主科学反对专制愚昧提倡新道德反对旧道德提倡新文学反对旧文学那再有可能就是促进了马克思主义的传播不过这些概念呢往往容易让人把一段历史理解成静止的了仿佛开始时是这些结束时还是这些但我们都知道历史它是流动的它不是静止的消化运动本身在这七年间一直在变化的像电视剧当中比较明显的一个变化就是陈独秀和胡适的关系早期亲的快穿一条裤子了那最后就变得渐行渐远但这些变化呀还只是河水表面一眼就能望见的我们深挖一层其实新文化运动的底层是有三个阶段的变化逻辑的我们一个个来讲首先第一个阶段最明显的就是从孔教束缚到个性解放那这一阶段的新文化运动本质上其实应该说是在给辛亥革命补
        问题：新文化运动的阶段

        输出示例1:
        "{
            1: {
                    'relation': , //相关性浮点数
                    'sentence': , //文本相关部分的第一句，长度为30-50字，以省略号结尾，例如"觉醒年代背后的历史脉络..."
            },
            2: {
            
            },
            ...
        }"
        输入示例2: 
        文本：三观正直一把尺扎扎实实讲历史我是李正欢迎收看正直讲史咱们这期聊聊今年的现象级历史剧觉醒年代那毫无疑问这部剧是近几年难得的优秀历史剧我自己在看的时候就想着一定要出一期视频来和大家聊一聊那现在b站中关于觉醒年代的视频非常多了啊有剧情解说也有那种高燃的混解不过身为一名历史老师我还是想从历史专业的角度和大家聊一聊觉醒年代背后的历史脉络首先觉醒年代的时间跨度首从一九一五年的新文化运动到一九二一年中共建党一共七年时间按理说这段历史其实是一个冷门的阶段因为我们之前往往更关注的都是党诞生后的故事很少关注党诞生前发生了什么事就比如现在如果我们让一个人形容一下新文化运动那他会说什么呢我猜可能比较多的还是高中课本上那些知识点像北大新青年然后各种代表人物再有就是经典的三提倡三反对提倡民主科学反对专制愚昧提倡新道德反对旧道德提倡新文学反对旧文学那再有可能就是促进了马克思主义的传播不过这些概念呢往往容易让人把一段历史理解成静止的了仿佛开始时是这些结束时还是这些但我们都知道历史它是流动的它不是静止的消化运动本身在这七年间一直在变化的像电视剧当中比较明显的一个变化就是陈独秀和胡适的关系早期亲的快穿一条裤子了那最后就变得渐行渐远但这些变化呀还只是河水表面一眼就能望见的我们深挖一层其实新文化运动的底层是有三个阶段的变化逻辑的我们一个个来讲首先第一个阶段最明显的就是从孔教束缚到个性解放那这一阶段的新文化运动本质上其实应该说是在给辛亥革命补
        问题：Amstrong公理在数据库中的应用

        输出示例2:
        "{
            1: {
                    'relation': , //相关性浮点数
                    'sentence': "", //不相关时为空白
            },
        }"
        """

        user_prompt = f"文本：{self.context.full_stripped_text}\n\
                        问题：{usrQuery}"
        
        # print(user_prompt)

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        
        self.queries['Vague Search'].append(messages)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={
                'type': 'json_object'
            }
        )

        res = json.loads(response.choices[0].message.content)
        # print(res)
        
        sent_timestamp_pairs = dict()
        for part_i in res:
            sent_timestamp_pairs[int(part_i)] = dict()
            sent_timestamp_pairs[int(part_i)]['sentence'] = res[part_i]['sentence']
            if len(res[part_i]['sentence']) == 0:
                break
            crntquery = res[part_i]['sentence'][:-3]
            sent_timestamp_pairs[int(part_i)]['timestamp'] = self.full_query_matching_retrieval(crntquery, from_vague_retriv=True)[crntquery]

        return sent_timestamp_pairs

    def export_queries(self, export_dir=None) -> None:
        # NOTE: to be tested.
        if export_dir is None:
            export_dir = self.context.intermediate_dir / 'queries.json'
        with open(export_dir, mode="a") as f:
            json.dump(self.queries, f, ensure_ascii=False)

