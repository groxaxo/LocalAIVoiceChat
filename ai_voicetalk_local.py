from RealtimeTTS import TextToAudioStream, CoquiEngine
from RealtimeSTT import AudioToTextRecorder
import llama_cpp
import torch
import json
import os

output = ""
llama_cpp_cuda = None
history = []
model = None
chat_params = {}
completion_params = {}
creation_params = {}
coqui_engine = None
stream = None
recorder = None
Llama = None

def load_dependencies():
    global llama_cpp_cuda, Llama
    if torch.cuda.is_available():
        try:
            print("Trying to import llama_cpp_cuda")
            import llama_cpp_cuda
        except ImportError:
            print("llama_cpp_cuda import failed")
            llama_cpp_cuda = None
    elif torch.version.hip:
        try:
            print("Trying to import llama_cpp")
            import llama_cpp
        except ImportError:
            print("ROCm is not available")
            llama_cpp = None

    def llama_cpp_lib():
        if llama_cpp_cuda is None:
            return llama_cpp
        else:
            return llama_cpp_cuda

    Llama = llama_cpp_lib().Llama

def load_config():
    global creation_params, completion_params, chat_params
    with open('creation_params.json') as f:
        creation_params = json.load(f)
    with open('completion_params.json') as f:
        completion_params = json.load(f)
    with open('chat_params.json') as f:
        chat_params = json.load(f)
    
    chat_params = replace_placeholders(chat_params, chat_params["char"], chat_params["user"])
    chat_params = replace_placeholders(chat_params, chat_params["char"], chat_params["user"], chat_params["scenario"])

    if 'logits_processor' not in completion_params or not completion_params['logits_processor']:
        completion_params['logits_processor'] = None

def initialize_models():
    global model, coqui_engine, stream, recorder
    print("Initializing LLM llama.cpp model ...")
    model = Llama(**creation_params)
    print("llama.cpp model initialized")

    print("Initializing TTS CoquiEngine ...")
    coqui_engine = CoquiEngine(language="en", speed=1.0)

    print("Initializing STT AudioToTextRecorder ...")
    stream = TextToAudioStream(coqui_engine, log_characters=True)
    recorder = AudioToTextRecorder(model=chat_params.get("stt_model", "tiny.en"), language="en", spinner=False)

def select_voice():
    while True:
        voice_number = input("Select voice (1-5): ")
        voice_path_wav = os.path.join("voices", f"voice{voice_number}.wav")
        voice_path_json = os.path.join("voices", f"voice{voice_number}.json")

        voice_path = None
        if os.path.exists(voice_path_wav):
            voice_path = voice_path_wav
        elif os.path.exists(voice_path_json):
            voice_path = voice_path_json

        if voice_path:
            coqui_engine.set_voice(voice_path)
            stream.feed(f"This is how voice number {voice_number} sounds like").play()
            accept_voice = input("Accept voice (y/n): ")
            if accept_voice.lower() != "n":
                break
        else:
            print(f"Voice file for voice {voice_number} not found. Please provide voice{voice_number}.wav or voice{voice_number}.json in the 'voices' directory.")

def replace_placeholders(params, char, user, scenario=""):
    for key in params:
        if isinstance(params[key], str):
            params[key] = params[key].replace("{char}", char)
            params[key] = params[key].replace("{user}", user)
            if scenario:
                params[key] = params[key].replace("{scenario}", scenario)
    return params

def write_file(file_path, content, mode='w'):
    with open(file_path, mode) as f:
        f.write(content)

def clear_console():
    os.system('clear' if os.name == 'posix' else 'cls')

def encode(string):
    return model.tokenize(string.encode() if isinstance(string, str) else string)

def count_tokens(string):
    return len(encode(string))

def create_prompt():
    prompt = f'<|system|>\n{chat_params["system_prompt"]}</s>\n'
    if chat_params["initial_message"]:
        prompt += f"<|assistant|>\n{chat_params['initial_message']}</s>\n"
    return prompt + "".join(history) + "<|assistant|>"

def generate():
    global output
    output = ""
    prompt = create_prompt()
    write_file('last_prompt.txt', prompt)
    completion_params['prompt'] = prompt
    first_chunk = True
    for completion_chunk in model.create_completion(**completion_params):
        text = completion_chunk['choices'][0]['text']
        if first_chunk and text.isspace():
            continue
        first_chunk = False
        output += text
        yield text

def chat_loop():
    clear_console()
    print(f'Scenario: {chat_params["scenario"]}\n\n')

    while True:
        print(f'>>> {chat_params["user"]}: ', end="", flush=True)
        user_text = recorder.text()
        print(f'{user_text}\n<<< {chat_params["char"]}: ', end="", flush=True)
        history.append(f"<|user|>\n{user_text}</s>\n")

        tokens_history = count_tokens(create_prompt())
        while tokens_history > 8192 - 500:
            history.pop(0)
            history.pop(0)
            tokens_history = count_tokens(create_prompt())

        generator = generate()
        stream.feed(generator)
        stream.play(fast_sentence_fragment=True, buffer_threshold_seconds=999, minimum_sentence_length=18, log_synthesized_text=True)
        history.append(f"<|assistant|>\n{output}</s>\n")
        write_file('last_prompt.txt', create_prompt())

def main():
    load_dependencies()
    load_config()
    initialize_models()
    select_voice()
    chat_loop()

if __name__ == '__main__':
    main()