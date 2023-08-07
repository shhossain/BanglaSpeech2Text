import argparse
from mimetypes import guess_type


def is_audio_file(filename):
    return guess_type(filename)[0].startswith("audio")


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
    parser.add_argument("-c", "--cache", type=str, help="cache directory")
    parser.add_argument("-o", "--output", type=str, help="output directory")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument(
        "-s", "--split", action="store_true", help="split audio file", default=False
    )
    parser.add_argument(
        "-sm", "--min_silence_length", type=int, help="min_silence_length", default=500
    )
    parser.add_argument(
        "-st", "--silence_thresh", type=int, help="silence_thresh", default=-16
    )
    parser.add_argument("-sp", "--padding", type=int, help="padding", default=300)
    parser.add_argument("--list", action="store_true", help="list of available models")
    parser.add_argument("--info", action="store_true", help="show model info")

    args = parser.parse_args()

    from banglaspeech2text.utils import nice_model_list

    if args.list:
        print(
            f"{nice_model_list()}\n\nFor more models, visit https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=bn&sort=likes"
        )
        return

    from banglaspeech2text.speech2text import Speech2Text, Model

    if args.info:
        model = Model(
            args.model,
            args.cache,
            load_pieline=False,
        )
        print(model)
        return

    if not args.input:
        parser.print_help()
        return

    sst = Speech2Text(args.model, args.cache, args.gpu)

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
        output += sst.recognize(
            filename,
            split=args.split,
            min_silence_length=args.min_silence_length,
            silence_thresh=args.silence_thresh,
            padding=args.padding,
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
