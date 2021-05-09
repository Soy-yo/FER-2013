import argparse
import sys

from emotion_detector import EmotionDetector


def main():

    parser = argparse.ArgumentParser(prog="detect_emotions",
                                     description="Detect emotions in images")
    
    parser.add_argument('paths', nargs='+',
                        help="path(s) to image(s) or folder(s) containing images")
    parser.add_argument('-m', '--model', choices=['vggface_knn', 'vggface', 'scratch'],
                        default='vggface_knn',  help="don't display results")
    parser.add_argument('-n', '--no-show', action='store_true', default=False,
                        help="don't display results")
    parser.add_argument('-s', '--save', default=None,
                        help="save results in this directory")
    
    args = parser.parse_args(sys.argv[1:])
    
    files = args.paths
    model = args.model
    show = not args.no_show
    save = args.save
    
    detector = EmotionDetector(model)
    results = detector.detect_emotion(files, show=show, save=save, verbose=True)


if __name__ == '__main__':
    main()