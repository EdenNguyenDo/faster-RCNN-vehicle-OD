from infer_script import infer
from ocsort_tracker.args import make_parser
from track_script import run_track


def main():
    args = make_parser("track_config.yaml").parse_args()
    if args.video_track:
        print("Running video tracking... \n")
        infer(args)
    else:
        print("Running detection files tracking... \n")
        run_track(args)

if __name__ == '__main__':
    main()