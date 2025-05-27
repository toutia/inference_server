import threading
import time
import triton_python_backend_utils as pb_utils


from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np
import torchaudio
import torch
from TTS.api import TTS

from transformers import AutoTokenizer


class TritonPythonModel:

    def initialize(self, args):
        """
        Called once when the model is loaded. Load your model here.
        """

        self.config = XttsConfig()
        self.config.load_json("/data/models/xtts/1/xtts_model/config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_dir="/data/models/xtts/1/xtts_model/",
        )
        self.model.cuda()
        # Precompute speaker embedding

        self.gpt_cond_latent, self.speaker_embedding = (
            self.model.get_conditioning_latents(
                audio_path=["/data/models/xtts/1/xtts_model/speaker.wav"]
            )
        )
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        """
        Called for every request. Uses the loaded model.
        """

        if len(requests) != 1:
            raise pb_utils.TritonModelException(
                "unsupported batch size " + len(requests)
            )
        request = requests[0]
        text = (
            pb_utils.get_input_tensor_by_name(request, "text")
            .as_numpy()[0]
            .decode("utf-8")
        )

        language = (
            pb_utils.get_input_tensor_by_name(request, "language")
            .as_numpy()[0]
            .decode()
        )

        # Start a separate thread to send the responses for the request. The
        # sending back the responses is delegated to this thread.
        thread = threading.Thread(
            target=self.response_thread,
            args=(request.get_response_sender(), text, language),
        )

        # A model using decoupled transaction policy is not required to send all
        # responses for the current request before returning from the execute.
        # To demonstrate the flexibility of the decoupled API, we are running
        # response thread entirely independent of the execute thread.
        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

    def response_thread(self, response_sender, text, language):

        chunks = self.model.inference_stream(
            text,
            language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            # language=language,
        )
        for chunk in chunks:
            chunk_np = (
                chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else chunk
            )
            out_tensor = pb_utils.Tensor("audio", chunk_np)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            response_sender.send(response)

        # We must close the response sender to indicate to Triton that we are
        # done sending responses for the corresponding request. We can't use the
        # response sender after closing it. The response sender is closed by
        # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1
