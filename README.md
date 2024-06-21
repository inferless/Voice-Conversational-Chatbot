# Voice-Conversational-Chatbot
- Welcome to an immersive tutorial crafted to guide you through the development of a voice conversational chatbot application, leveraging state-of-the-art serverless technologies. Throughout this tutorial, you’ll gain insights into seamlessly integrating multiple models within Inferless to construct a robust application.
- We will leverage the Bark, Faster-Whisper, Transformers,and Inferless, a cutting-edge platform for deploying and serving machine learning models efficiently.
## Architecture
![Architecture Diagram](https://i.postimg.cc/Z5rsygMC/voice-chatbot.png)
---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the **Create new Runtime** button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and use the forked repo URL as the **Model URL**.

After the create model step, while setting the configuration for the model make sure to select the appropriate runtime.

### ADD your HF token as an Env Variable during import ( Step 4 ) :

Add 

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/github-custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.

```bash
curl --location '<your_inference_url>' \
          --header 'Content-Type: application/json' \
          --header 'Authorization: Bearer <your_api_key>' \
          --data '{
  "inputs": [
              {
                "data": [
                   "<audio_base64>"
                ],
                "name": "audio_base64",
                "shape": [
                  1
                ],
                "datatype": "BYTES"
              }
            ]
          }
   '
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](#input) for more.

```python
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
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting `self.pipe = None`.


For more information refer to the [Inferless docs](https://docs.inferless.com/).
