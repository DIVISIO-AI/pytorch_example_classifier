import os
import argparse
from pathlib import Path
from train import build_model, read_hyperparameters, create_loader
import torch
import shutil


class Tester:
    """Class to run a test with the result of a trainer, saves
    predictions into a target folder for better debugging training
    results."""

    def __init__(self, args):
        # load configuration
        self.h_params, _ = read_hyperparameters(args.config)
        # build model and load checkpoint
        self.model = build_model(
            self.h_params.architecture,
            self.h_params.model_weights,
            self.h_params.class_count
        )
        self.model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        self.model.eval()
        # make sure output folders exist
        self.output = Path(args.output)
        print(f'Writing debug images to {self.output}')
        os.makedirs(self.output, exist_ok=True)

    def save_predicted_image(self, original_path, label, pred):
        original_path = os.path.normpath(original_path)
        path_parts = original_path.split(os.sep)
        # create output subfolder for the label
        if not os.path.exists(os.path.join(self.output, label)):
            os.makedirs(os.path.join(self.output, label))
        target_path = os.path.join(self.output, label,
                                   f'{pred}_'
                                   f'{str(path_parts[-1]).lower()}')
        shutil.copy(original_path, target_path)

    def test(self, input_folder):
        _, dataset = create_loader(
            image_folder=input_folder,
            image_size=self.h_params.image_size,
            batch_size=self.h_params.batch_size,
            train=False)
        count_true = 0
        count_items = 0

        for idx in range(0, dataset.__len__()):
            filepath = dataset.imgs[idx][0]
            (input, label) = dataset.__getitem__(idx)
            input_batch = input.unsqueeze(0)
            logits = self.model(input_batch)
            pred = logits.max(dim=1)[1].item()
            is_pred_correct = pred == label
            if is_pred_correct:
                count_true += 1
            count_items += 1
            self.save_predicted_image(filepath, dataset.classes[label], dataset.classes[pred])

        accuracy = count_true / count_items
        print(f'accuracy: {accuracy}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file (required)")
    parser.add_argument("-cp", "--checkpoint", help="Pretrained classification model")
    parser.add_argument("-o", "--output", help="output debug folder")
    parser.add_argument("input", help="folder with test data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.input):
        print(f'{args.input} this is not a folder')
        exit()

    tester = Tester(args)
    tester.test(args.input)
