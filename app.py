from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from bark import SAMPLE_RATE, generate_audio, preload_models
import numpy as np
import io
import base64
import soundfile as sf
import nltk

class InferlessPythonModel:
    def initialize(self):
        # Load speech to text model
        self.audio_file = "output.mp3"
        model_size = "large-v3"
        self.model_whisper = WhisperModel(model_size, device="cuda", compute_type="float16")
        
        # Load Mistral instruct, text to text model
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.model_mistral = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load Bark, Text to Speech
        self.SPEAKER = "v2/en_speaker_6"
        preload_models()

        # Download nltk punkt
        nltk.download('punkt')
       
    def base64_to_mp3(self, base64_data, output_file_path):
        # Convert base64 audio data to mp3 file
        mp3_data = base64.b64decode(base64_data)
        with open(output_file_path, "wb") as mp3_file:
            mp3_file.write(mp3_data)

    def infer(self, inputs):
        audio_data = inputs["audio_base64"]
        self.base64_to_mp3(audio_data, self.audio_file)
        
        # Transcribe audio to text
        segments, info = self.model_whisper.transcribe(self.audio_file, beam_size=5)
        user_text = ''.join([segment.text for segment in segments])
        
        # Generate prompt for Mistral model
        messages = [{"role": "user", "content": f"You are a helpful, respectful and honest assistant. Answer the following question in exactly in few words from the context. {user_text}"}]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        model_inputs = encodeds.to("cuda")
        
        # Generate text response using Mistral model
        generated_ids = self.model_mistral.generate(model_inputs, max_new_tokens=80, do_sample=True)
        generated_text = self.tokenizer.batch_decode(generated_ids[:, encodeds.shape[1]:], skip_special_tokens=True)[0]
        
        # Process generated text into audio
        script = generated_text.replace("\n", " ").strip()
        sentences = nltk.sent_tokenize(script)
        
        silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
        
        pieces = []
        
        for sentence in sentences:
            audio_array = generate_audio(sentence, history_prompt=self.SPEAKER)
            pieces += [audio_array, silence.copy()]
        
        # Convert audio pieces into base64
        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(pieces), SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
         
        return {"generated_audio_base64": base64_audio}

    def finalize(self):
        # Finalize resources if needed
        pass
