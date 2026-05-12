import time
import app as gradv_app
ui = gradv_app.build_app()
ret = ui.launch(server_name='0.0.0.0', server_port=7860, share=True, inbrowser=False, show_error=True, prevent_thread_lock=True, quiet=False)
print('LOCAL_URL=' + str(ret[1]), flush=True)
print('SHARE_URL=' + str(ret[2]), flush=True)
while True:
    time.sleep(3600)
