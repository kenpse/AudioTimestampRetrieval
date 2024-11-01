

## 一个音频的语音时间戳检索工具

用阿里开源的funasr-Paraformer模型实现的ASR和语音时间戳工具

用到的库见requirements.txt，主要是pytorch, funasr；用gradio写了个小的UI，Selenium写了个小的b站音频爬虫（audio_crawler.py里）

爬下来的原始音频会放在data/raw里；原始音频会被切成1分钟的片段，然后形成一个文件夹放在data/intermediate里。中间结果和最后跑出来的asr+时间戳（每个片段的inference结果会打包成json）也会放到这个文件夹

全字检索就是简单的字符串匹配（正则）；用deepseek的API写了一个模糊检索的功能（prompt LLM从全文里提取和query相关的句子）

##### 注意
1. funasr有自己的时间戳工具，在https://github.com/modelscope/FunASR/blob/main/funasr/utils/timestamp_tools.py
2. 现在时间戳对纯中文（纯英没有试过）已经识别的比较准确了，但是中英混杂会不准(token数量对不上)，音频切片能把这种misalignment限定在片段长度里

