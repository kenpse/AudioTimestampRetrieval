import ffmpeg
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from bs4 import BeautifulSoup
import json
import requests
import time

# Configurations
# NOTE: can and should be replaced by one's own browser configurations
edge_binary_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
custom_browserSim_headers_NO1 = {
    "referer": "https://www.bilibili.com",
    "origin": "https://www.bilibili.com",
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36',
}


# 1. bilibili.com audio crawling process
def procdef_download_audio_by_bvid(bv: str, out_filepath: str, quality=2) -> None:
    # NOTE: Very rudimentary implementation of bilibili audio retrieval function with no multi-thread implementation
    # STEP 1. create edge webdriver
    edgeops = EdgeOptions()
    edgeops.binary_location = edge_binary_path
    edgedriver = webdriver.Edge(options=edgeops)

    # STEP 2. crawl raw content from webpage
    bili_video_urlstem = r"https://www.bilibili.com/video"
    bvpageurl = bili_video_urlstem + '/' + bv

    edgedriver.get(bvpageurl)
    souped_src = BeautifulSoup(edgedriver.page_source, 'html.parser')

    tag_script_elems = souped_src.find_all('script')
    tag_script_window_playInfo_txt = tag_script_elems[3].get_text()                     # TODO: find a way to determine the index of tag_script_elems containing wwindow.__playinfo__ for higher robustness
    tag_script_window_playInfo_json = json.loads(tag_script_window_playInfo_txt[20:])   # TODO: change hardwired :20 to strip window.__playinfo__
    tag_script_window_playInfo_data_entry = tag_script_window_playInfo_json['data']
    tag_script_window_playInfo_data_dash = tag_script_window_playInfo_data_entry['dash']
    
    audio_info = tag_script_window_playInfo_data_dash['audio']
    audio_urls = audio_info[quality]
    audio_base_url = audio_urls['base_url']

    time.sleep(3)

    try:
        res_content = requests.get(url=audio_base_url, 
                         headers=custom_browserSim_headers_NO1, 
                         stream=True
                        )
        res_content.raise_for_status()

    except Exception as e:
        print(e)

    res_content = res_content.content

    ffmpeg_cmd = ( # <class 'subprocess.Popen'>
        ffmpeg
        .input('pipe:')
        .output(filename=out_filepath
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )
    # ffmpeg.input('pipe:').output(filename=out_filepath).overwrite_output().run_async(pipe_stdin=True, pipe_stderr=True).run(input=res_content)
    ffmpeg_cmd.communicate(input=res_content)
    ffmpeg_cmd.kill()



