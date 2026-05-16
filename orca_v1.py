import time
import threading
import webbrowser
from pathlib import Path

import server

#recording mode — WAV_PATH > local mic.
server.ASR_MODEL = 'large-v3'
server.LLM_MODEL = 'llama3.1:8b'
server.TEMPLATE_PATH = 'report_template_1.json'
server.WAV_PATH = 'recordings/recording_20260315_182554.wav'


def _open_browser():

    '''
    opens the ORCA UI in the default browser after a short delay.
    '''

    time.sleep(1)
    html_path = Path(__file__).parent / 'orca_ui.html'
    webbrowser.open(html_path.as_uri())


#open the ui in the default browser after a short delay...
threading.Thread(target = _open_browser, daemon = True).start()

#start the flask backend...
print('ORCA backend running on http://localhost:5050')
server.app.run(port = 5050, debug = False)