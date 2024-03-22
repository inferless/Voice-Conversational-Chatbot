from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from bark import SAMPLE_RATE, generate_audio, preload_models
import time
import numpy as np
import io
import base64
import soundfile as sf


class InferlessPythonModel:  
    def initialize(self):
        # load speech to text model
        model_size = "large-v3"
        self.model_whisper = WhisperModel(model_size, device="cuda", compute_type="float16")
        
        # Load Mistral instruct, text to text model
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Specify the model repository ID
        # Define sampling parameters for model generation
        # self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)
        # Initialize the LLM object
        # self.llm = LLM(model=model_id)
        self.model_mistral = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load Bark, Text to Speech
        preload_models()
    
    def base64_to_mp3(self, base64_data, output_file_path):
        mp3_data = base64.b64decode(base64_data)
        with open(output_file_path, "wb") as mp3_file:
                mp3_file.write(mp3_data)

    def infer(self, inputs):
        audio_data = inputs["audio_base64"]
        audio_file = "output.mp3"
        self.base64_to_mp3(audio_data,audio_file)
        segments, info = self.model_whisper.transcribe(audio_file, beam_size=5)
        user_text = ''.join([segment.text for segment in segments])
        
        
        inputs = self.tokenizer(user_text, return_tensors="pt").to("cuda")
        outputs = self.model_mistral.generate(**inputs, max_new_tokens=200)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)


        audio_array = generate_audio(generated_text)
        buffer = io.BytesIO()
        sf.write(buffer, audio_array,SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {"user_question":user_text,"bot_answer":generated_text,"generated_audio_base64": base64_audio}
    def finalize(self):
        pass
