import gradio as gr, time
with gr.Blocks() as demo:
    gr.Markdown('# test')
ret = demo.launch(server_name='0.0.0.0', server_port=7862, share=True, prevent_thread_lock=True)
print('SHARE_URL=' + str(ret[2]), flush=True)
while True:
    time.sleep(3600)
