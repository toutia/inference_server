###############################" asr #########################################

before using record.py to record an audio set system default input to the right micro 

python record.py

streaming 
python  riva_streaming_asr_client.py      --input-file=output.wav

non streaming 

python   transcribe_file.py --input-file=output.wav 

###############################"  transcribe mic #######################################


--list-devices to get the input device 


python  transcribe_mic.py --input-device=0




################################" tts ###################################################"""

list voices 
 ~/dev/triton_manager/examples$ python   talk.py --list-voices



 python   talk.py --text="hello this is a  test "   --voice="English-US.Female-1"   --play-audio  --stream


if   =>  reinstall pyaudio

###########################################################################################
Generating audio for request...
Time to first audio: 3.777s
Traceback (most recent call last):
  File "/home/touti/dev/triton_manager/examples/talk.py", line 144, in <module>
    main()
  File "/home/touti/dev/triton_manager/examples/talk.py", line 122, in main
    sound_stream(resp.audio)
  File "/home/touti/dev/triton_manager/.venv_tm/lib/python3.10/site-packages/riva/client/audio_io.py", line 132, in __call__
    self.stream.write(audio_data)
  File "/home/touti/dev/triton_manager/.venv_tm/lib/python3.10/site-packages/pyaudio.py", line 586, in write
    pa.write_stream(self._stream, frames, num_frames,
SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
##########################################################################################

pip uninstall pyaudio 


sudo apt-get install portaudio19-dev
pip install pyaudio





############################################### NLP punctuation  #########################################


(.venv_tm) touti@ubuntu ~/dev/triton_manager/examples$ python  punctuation_client.py --interactive 
Enter a query: will you have $103 and ₩111 at 12:45 pm
Inference complete in 34.6107 ms
Will you have $103 and ₩111 at 12:45 pm?


