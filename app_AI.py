from infer_script import infer
from ocsort_tracker.args import make_parser
from track_script import run_track


def main():
    # allows testing of different config files in one run
    config_file_paths = ["track_config.yaml"]
    for config_file in config_file_paths:
        args = make_parser(config_file).parse_args()
        if args.video_track:
            print(f"Running video tracking with config file {config_file}... \n")
            infer(args)
        else:
            #todo check multi-config file tracking
            print(f"Running detection files tracking with config file {config_file}... \n")
            run_track(args)

if __name__ == '__main__':
    main()