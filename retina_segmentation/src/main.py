import argparse
from retina_segmentation.src import train, evaluate, visualize_results, export_model

def main():
    parser = argparse.ArgumentParser(description="DeepVessel-Net: Retina Vessel Segmentation")
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'visualize', 'export'],
                        help="Mode to run: train / evaluate / visualize / export")

    parser.add_argument('--config', type=str, default='config.yaml',
                        help="Optional config file path (if used)")

    args = parser.parse_args()

    if args.mode == 'train':
        train.main()
    elif args.mode == 'evaluate':
        evaluate.main()
    elif args.mode == 'visualize':
        visualize_results.main()
    elif args.mode == 'export':
        export_model.main()
    else:
        raise ValueError("Invalid mode selected.")

if __name__ == '__main__':
    main()
