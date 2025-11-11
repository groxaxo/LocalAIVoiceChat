if __name__ == '__main__':
    from RealtimeTTS import TextToAudioStream, CoquiEngine
    from RealtimeSTT import AudioToTextRecorder
    import llama_cpp
    import torch
    import json
    import os
    from hardware_detector import get_optimal_config, load_optimal_config, print_hardware_info

    output = ""
    llama_cpp_cuda = None

    # Detect hardware capabilities
    print("\n" + "="*60)
    print("Detecting hardware for optimal performance...")
    print("="*60)
    hw_config = print_hardware_info()

    if torch.cuda.is_available():
        try:
            print (f"\nTrying to import llama_cpp_cuda...")
            import llama_cpp_cuda
            print("llama_cpp_cuda imported successfully")
        except:
            print (f"llama_cpp_cuda import failed, falling back to llama_cpp")
            llama_cpp_cuda = None
    elif torch.version.hip:
        try:
            print (f"Trying to import llama_cpp with ROCm support...")
            import llama_cpp
        except:
            print (f"ROCm is not available")
            llama_cpp = None

    def llama_cpp_lib():
        if llama_cpp_cuda is None:
            print ("Using: llama_cpp (CPU or standard CUDA)")
            return llama_cpp
        else:
            print ("Using: llama_cpp_cuda (optimized CUDA)")
            return llama_cpp_cuda

    Llama = llama_cpp_lib().Llama

    history = []


    def replace_placeholders(params, char, user, scenario = ""):
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

    # Load optimized configuration based on hardware detection
    creation_params = load_optimal_config()
    
    # Allow manual override if specific config file exists
    if os.path.exists('creation_params_override.json'):
        print("\nFound creation_params_override.json, using manual configuration")
        with open('creation_params_override.json') as f:
            creation_params = json.load(f)
    
    with open('completion_params.json') as f:
        completion_params = json.load(f)
    with open('chat_params.json') as f:
        chat_params = json.load(f)
    
    chat_params = replace_placeholders(chat_params, chat_params["char"], chat_params["user"])
    chat_params = replace_placeholders(chat_params, chat_params["char"], chat_params["user"], chat_params["scenario"])

    if not completion_params['logits_processor']:
        completion_params['logits_processor'] = None

    # Display loaded configuration
    print(f"\nModel Configuration:")
    print(f"  GPU Layers: {creation_params.get('n_gpu_layers', 0)}")
    print(f"  CPU Threads: {creation_params.get('n_threads', 4)}")
    print(f"  Batch Size: {creation_params.get('n_batch', 512)}")
    print(f"  Context Size: {creation_params.get('n_ctx', 8192)}")

    # Initialize AI Model
    print("\nInitializing LLM llama.cpp model ...")
    model = Llama(**creation_params)
    print("llama.cpp model initialized successfully")


    print("\nInitializing TTS CoquiEngine ...")    
    # import logging
    # logging.basicConfig(format='AI Voicetalk: %(message)s', level=logging.DEBUG)
    # coqui_engine = CoquiEngine(cloning_reference_wav="female.wav", language="en", level=logging.DEBUG)
    
    # Optimize TTS engine settings for lower latency
    tts_params = {
        "cloning_reference_wav": "female.wav",
        "language": "en",
        "speed": 1.1  # Slightly faster for reduced latency
    }
    
    # GPU-specific optimizations for TTS
    if hw_config['gpu']['available']:
        print("GPU detected: Enabling GPU acceleration for TTS")
        # CoquiEngine will automatically use GPU if available
    
    coqui_engine = CoquiEngine(**tts_params)
    print("CoquiEngine initialized successfully")

    print("\nInitializing STT AudioToTextRecorder ...")
    
    # Optimize STT settings based on hardware
    stt_model = "tiny.en"  # Fast, low latency
    if hw_config['gpu']['available'] and hw_config['gpu']['is_ampere']:
        # Ampere GPUs can handle larger models efficiently
        stt_model = "base.en"  # Better accuracy with acceptable latency on Ampere
        print(f"Ampere GPU detected: Using '{stt_model}' model for better accuracy")
    else:
        print(f"Using '{stt_model}' model for optimal speed")
    
    stream = TextToAudioStream(coqui_engine, log_characters=True)
    recorder = AudioToTextRecorder(
        model=stt_model,
        language="en",
        spinner=False,
        silero_sensitivity=0.4,  # Slightly more aggressive voice detection
        webrtc_sensitivity=3,    # Balanced sensitivity
        post_speech_silence_duration=0.4  # Reduced for faster response
    )


    print()
    while True:
        voice_number = input(f"Select voice (1-5): ")
        voice_path = os.path.join("voices", f"voice{voice_number}.wav")
        coqui_engine.set_voice(voice_path)

        stream.feed(f"This is how voice number {voice_number} sounds like").play()
        #stream.feed("This is how your selected voice sounds like").play()
        accept_voice = input(f"Accept voice (y/n): ")
        if accept_voice.lower() != "n":
            break


    clear_console()
    print(f'Scenario: {chat_params["scenario"]}\n\n')

    while True:
        print(f'>>> {chat_params["user"]}: ', end="", flush=True)
        print(f'{(user_text := recorder.text())}\n<<< {chat_params["char"]}: ', end="", flush=True)
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