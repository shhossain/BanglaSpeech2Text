import argparse
from mimetypes import guess_type

from banglaspeech2text.utils.models import nice_model_list
from banglaspeech2text.utils.loading import LoadingIndicator


def is_audio_file(filename):
    gt = guess_type(filename)
    if gt[0] is None:
        return False
    return gt[0].startswith("audio")


def use_mic(stt):
    import speech_recognition as sr  # type: ignore

    r = sr.Recognizer()
    while True:
        try:
            with sr.Microphone() as source:
                print("Say something!")
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                with LoadingIndicator("Recognizing"):
                    output = stt.recognize(audio)
                print(output)
        except KeyboardInterrupt:
            print("Exiting...")
            exit()
        except Exception as e:
            print(e)
            continue


def main():
    parser = argparse.ArgumentParser(description="Bangla Speech to Text")
    parser.add_argument(
        "input",
        metavar="INPUT",
        type=str,
        nargs="*",
        help="input file(s) or list of files",
    )
    parser.add_argument("-gpu", action="store_true", help="use gpu", default=False)
    parser.add_argument("-c", "--cache", type=str, help="cache directory", default=None)
    parser.add_argument("-o", "--output", type=str, help="output directory")
    parser.add_argument("-m", "--model", type=str, help="model name", default="base")
    parser.add_argument("-sp", "--padding", type=int, help="padding", default=300)
    parser.add_argument("--list", action="store_true", help="list of available models")
    parser.add_argument("--info", action="store_true", help="show model info")
    parser.add_argument("--mic", action="store_true", help="use microphone")

    args = parser.parse_args()

    if args.list:
        print(
            f"{nice_model_list()}\n\nFor more models, visit https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=bn&sort=likes"
        )
        return

    from banglaspeech2text.speech2text import Speech2Text, ModelMetadata

    if args.info:
        model = ModelMetadata(
            args.model,
        )
        print(model)
        return

    if not args.input and not args.mic:
        parser.print_help()
        return

    sst = Speech2Text(args.model)

    if args.mic:
        use_mic(sst)
        return

    output = ""
    audio_files = []

    for filename in args.input:
        if is_audio_file(filename):
            audio_files.append(filename)
        else:
            with open(filename, "r") as f:
                audio_files.extend([line.strip() for line in f])

    n = len(audio_files)
    for filename in audio_files:
        print(f"Recognizing {filename}...")
        with LoadingIndicator(f"Recognizing {filename}"):
            output += sst.recognize(
                filename,
            )
        if n > 1:
            output += "\n" + "=" * 50 + "\n"
            n -= 1

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
