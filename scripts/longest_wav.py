import os, sys, torchaudio

def main():
    if len(sys.argv) != 2:
        print("Usage: python longest_file.py <audio_folder>")
        sys.exit(1)

    audio_folder = sys.argv[1]

    if not os.path.isdir(audio_folder):
        print("No file was found at the given path.")
        sys.exit(1)

    max_samples = 0
    max_file = ""
    max_seconds = 0

    for file in os.listdir(audio_folder):
        if file.endswith(".wav"):
            audio_path = os.path.join(audio_folder, file)
            audio, sr = torchaudio.load(audio_path)

            samples = audio.shape[-1]

            if samples > max_samples:
                max_samples = samples
                max_file = file
                max_seconds = samples / float(sr)

    print(f"The longest file in {audio_folder} is {max_file} with {max_samples} samples ({max_seconds} seconds).")

if __name__ == "__main__":
    main()