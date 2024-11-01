# coding=utf-8

import gradio as gr

from logic import AudioFileTranscript, QueryRetrieval
from logic import audio_dir, Path
from audio_crawler import procdef_download_audio_by_bvid

def retrieve_proc(audio_filepath, mode, query):
	print("input: ", audio_filepath)
	if len(audio_filepath.split('/')) == 1 and len(audio_filepath.split('.')) == 1:
		# given bvid
		bv, filename = audio_filepath, audio_filepath + '_audio.wav'
		audio_filepath = audio_dir / filename
		print("processed path: ", audio_filepath)
		if not Path(audio_filepath).exists():
			procdef_download_audio_by_bvid(bv, audio_filepath)

	transAudio = AudioFileTranscript(audio_filepath)
	audioQuery = QueryRetrieval(transAudio)
    
	if mode == "Full Query Matching":
		res = audioQuery.full_query_matching_retrieval(query=query, from_vague_retriv=False)
	else:
		res = audioQuery.vague_query_retrieval(query)
	# audioQuery.export_queries()
    
	return res


def launch():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    audio_inputs = gr.Audio(label="Display audio file here.")
                    audio_filepath = gr.Textbox(label="Local Audio Filepath")
                
                query_inputs = gr.Textbox(label="Query")

                with gr.Accordion("Configuration"):
                    mode_inputs = gr.Dropdown(choices=["Full Query Matching", "Vague Search"],
												  value="Full Query Matching",
												  label="Mode")
                fn_button = gr.Button("Start", variant="primary")
				
                text_outputs = gr.Textbox(label="Results")
             
        fn_button.click(retrieve_proc, inputs=[audio_filepath, mode_inputs, query_inputs], outputs=text_outputs)

    demo.launch()


if __name__ == "__main__":
    # iface.launch()
    
	# test_video = "BV1Pyy6Y3EHB"

    launch()

