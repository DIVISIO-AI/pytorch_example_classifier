import argparse
import datetime
import logging
import os.path
import mlflow
import shutil
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------- helpers --------------------------------

class HyperParams:
    def __init__(self, yaml_config, yaml_filepath):
        # run_name extracted from yaml filename: bar/baz/foo.yaml -> run_name = "foo"
        yaml_filename = os.path.basename(yaml_filepath)
        # the name of the run is determined by the config file name - that way, we immediately know
        # where the results of a run are
        self.run_name = yaml_filename.rpartition('.')[0]
        # The overall experiment name
        self.experiment_name = yaml_config.get("experiment")
        # Definition of training set(s)
        self.training_folder = yaml_config.get("trainingFolder")
        # Definition of validation set(s)
        self.validation_folder = yaml_config.get("validationFolder")
        # Number of epochs to train for
        self.num_epochs = yaml_config.get("numEpochs")
        # Effective batch size
        self.batch_size = yaml_config.get("batchSize")
        # Overall architecture to use, e.g. resnet50
        self.architecture = yaml_config.get("architecture")
        # input image size for the architecture
        self.image_size = yaml_config.get("imageSize")
        # number of classes to classify
        self.class_count = yaml_config.get("classCount")
        # The pretrained weights to use, may be empty for training with a freshly initialized net
        self.model_weights = yaml_config.get("modelWeights")
        # The initial learning rate - adjusted by a ReduceLROnPlateau Scheduler
        # (you may want to make the scheduler to use configurable as well)
        self.learning_rate = yaml_config.get("learningRate")
        # How often to log in seconds, -1 logs after every update
        self.log_every = yaml_config.get("logEvery")
        # How often to validate in seconds, -1 validates after every epoch
        self.validate_every = yaml_config.get("validateEvery")
        # How often to save in seconds, -1 saves after every epoch,
        # result is always saved at the end
        self.save_every = yaml_config.get("saveEvery")
        # The log level to use for the log file
        self.log_level = yaml_config.get("logLevel").upper()
        # last checkpoint to load, allows resuming of a training, may be empty
        self.load_checkpoint = yaml_config.get("checkpoint")


def read_hyperparameters(config_filepath):
    # load configuration
    yaml_config = None
    with open(config_filepath, "r") as stream:
        try:
            yaml_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # build hyperparams object from yams
    return HyperParams(yaml_config, config_filepath), yaml_config


def build_model(architecture, model_weights, class_count):
    """Builds the model used for training, by initializing an empty initial model for the
    given architecture and then optionally loading existing model weights"""
    # we only use existing architectures from the pytorch hub for this example - if you use
    # other sources, you need to move the model creation into one of the if clauses below
    model = torch.hub.load("pytorch/vision", architecture, weights=model_weights)
    # now we still need to adjust the loaded model to the number of classes we are dealing with
    # as all models have different names for their output layer or may even have a very different
    # structure, we need to do this on a per-architecture basis
    if architecture.startswith("vgg"):
        num_features = model.classifier[6].in_features
        # modify output layer
        model.classifier[6] = nn.Linear(num_features, class_count)
    elif architecture.startswith("resnet"):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, class_count)
    else:
        raise ValueError("Cannot adjust architecture " + architecture)

    return model


def create_loader(image_folder, image_size, batch_size, train):
    logging.info(f"Loading dataset from {image_folder}")
    transform = transforms.Compose([
        # Pre-process the image and convert into a tensor
        # (for advanced preprocessing check out Albumentations!)
        # we need to resize to the right input size for the model
        transforms.Resize(image_size),
        # make sure input is square
        transforms.CenterCrop(224),
        # turn input into a tensor so it can be processed by the network
        transforms.ToTensor(),
        # we need to normalize the input (this is often model specific, in this case it is necessary
        # for all pretrained torchvision models - you may have to make the data loading
        # configurable)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset with images in subfolders (this is a torch built-in, more complex datasets
    # may require your own custom class - maybe with its own configuration)
    data_set = datasets.ImageFolder(image_folder, transform=transform)

    classes = data_set.classes
    logging.info(f'Classes: {classes}')
    logging.info(f'Images: {len(data_set)}')

    return DataLoader(data_set, batch_size=batch_size, shuffle=train, num_workers=4), data_set

# ------------------------------ trainer -------------------------------


class Trainer:
    """ A trainer contains all the state during training. It is cleaner than keeping that state
     in global variables. """

    def __init__(self, args):
        # load configuration
        self.h_params, yaml_config = read_hyperparameters(args.config)
        # init state
        self.best_loss = 9999999  # best loss so far - we initialize this to a very high value
        self.model = None  # the model to train, will be initialized later
        self.optimizer = None  # the optimizer to use for training
        self.lr_scheduler = None  # the scheduler for the learning rate
        # (you generally do not want to keep the learning rate constant)
        self.loss_function = None  # loss function to use
        self.epoch = -1  # current epoch
        self.optimizer_step_counter = 0  # current optimizer step count, used to calculate averages
        self.last_save = time.time()  # when we last saved
        self.last_validate = time.time()  # when we last ran validation

        # make sure output path exists
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_path = Path(self.h_params.experiment_name) / self.h_params.run_name / now
        self.output_path.mkdir(parents=True, exist_ok=True)

        # configure logger
        log_file = self.output_path / "training.log"
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=self.h_params.log_level,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filemode='a',
                            filename=log_file
                            )
        print(f"Redirecting log output to {log_file}")
        logging.info(f"Using config file {args.config}")
        logging.info(f"Using output_path {self.output_path}")

        # determine device (we always use the GPU if available, to control which GPUs to use,
        # simply use the CUDA_VISIBLE_DEVICES environment variable)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

        # prepare mlflow tracking (optional, you can use other tracking frameworks or just
        # write csv files with the values you are interested in)
        mlflow.set_experiment(experiment_name=self.h_params.experiment_name)
        run = mlflow.start_run(run_name=self.h_params.run_name)
        logging.info(f"Current run ID is: {run.info.run_id}")
        logging.info(f"Current experiment is {self.h_params.experiment_name}")

        # log all non-list yaml params into mlflow
        for key, value in yaml_config.items():
            if not isinstance(value, list):
                mlflow.log_param(key, value)

        # make a copy of the config file, so we have the exact same config the training was
        # started with together with the results
        shutil.copy(Path(args.config), self.output_path / os.path.basename(args.config))

        logging.info("Intitializing train data loader.")
        self.training_data_loader, _ = create_loader(
            image_folder=self.h_params.training_folder,
            image_size=self.h_params.image_size,
            batch_size=self.h_params.batch_size,
            train=True)
        logging.info("Intitializing validation data loader.")
        self.validation_data_loader, _ = create_loader(
            image_folder=self.h_params.validation_folder,
            image_size=self.h_params.image_size,
            batch_size=self.h_params.batch_size,
            train=False)

    def initialize_model(self):
        """Initializes the model and all other entities we need for training."""
        # we move model building into a separate method
        # so it is also callable from the testing script
        model = build_model(
            self.h_params.architecture,
            self.h_params.model_weights,
            self.h_params.class_count
        )
        # load an existing checkpoint if necessary
        if self.h_params.load_checkpoint is not None:
            logging.info(f"starting from checkpoint {self.h_params.load_checkpoint}")
            model.load_state_dict(torch.load(self.h_params.load_checkpoint))
        else:
            logging.info("starting from scratch")

        # Set optimizer on parameters to update (you could make the optimizer configurable,
        # but in practice we use Adam so often that we just chose to hardcode it here)
        params_to_update = model.parameters()
        self.optimizer = optim.Adam(params_to_update, lr=self.h_params.learning_rate)
        # Initialize the learning rate scheduler - this is something you might want to make
        # configurable as well
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

        # Setup the loss fxn - we only train classifiers, so it is always softmax, for other
        # trainings you may want to make this configurable
        self.loss_function = nn.CrossEntropyLoss()

        # move the model to the target device - for multi device training, you might want to add
        # something like DataParallel here
        self.model = model.to(self.device)

    def train_epoch(self):
        accumulated_loss = 0
        sum_correct_predictions = 0
        sum_predictions = 0
        step_count_since_last_log = 1
        last_log = 0

        self.epoch += 1
        logging.info(f"starting training for epoch {self.epoch}")
        self.model.train()
        self.optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(self.training_data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            torch.set_grad_enabled(True)

            logits = self.model(inputs)

            loss = self.loss_function(logits, labels)

            accumulated_loss += loss.item()
            loss.backward()

            # accuracy calculation
            class_predictions = logits.max(dim=1)[1]
            sum_correct_predictions += torch.sum(class_predictions == labels).item()
            sum_predictions += logits.shape[0]

            # update the weights
            self.optimizer_step_counter += 1
            step_count_since_last_log += 1
            self.optimizer.step()
            self.optimizer.zero_grad()

            # log current losses if enough time has passed (do NOT do this after every step -
            # you will spend more time logging than training
            if time.time() - last_log > self.h_params.log_every:
                avg_loss = accumulated_loss / step_count_since_last_log
                accuracy = sum_correct_predictions / sum_predictions
                last_log = time.time()
                step_count_since_last_log = 0
                accumulated_loss = 0
                # multiply by batch split to get a comparable, non-averaged current value
                current_loss = loss.item()
                logging.info(f"[TRN] step {self.optimizer_step_counter}"
                             f"\ttrain loss = {avg_loss:.6f} "
                             f"\tcurrent loss = {current_loss:.6f} "
                             f"\ttrain acc = {accuracy:.6f}")
                mlflow.log_metric("train_loss", avg_loss, self.optimizer_step_counter)
                mlflow.log_metric("current_loss", current_loss, self.optimizer_step_counter)
                mlflow.log_metric("train_acc", accuracy, self.optimizer_step_counter)

            # check if we should validate after a micro batch has completed
            if 0 < self.h_params.validate_every < time.time() - self.last_validate:
                self.validate(self.validation_data_loader)
                self.last_validate = time.time()
                # switch model back to training mode
                self.model.train()

            # check if we should save after a micro batch has completed
            if 0 < self.h_params.save_every < time.time() - self.last_save:
                self.save_model()
                self.last_save = time.time()

        # save at the end of an epoch if configured to do so
        if self.h_params.save_every == -1:
            self.save_model()

        # validate at end of epoch if configured to do so
        if self.h_params.validate_every == -1:
            self.validate(self.validation_data_loader)

    def validate(self, validation_data_loader):
        """Performs validatin - what validation is depends very much on the model in training.
        An autoencoder may have no validation, a large language model might run > 100 ablation tests."""
        logging.info("doing validation step")
        validation_step_counter = 0
        accumulated_loss = 0
        sum_correct_predictions = 0
        sum_predictions = 0

        # we re-implement the loop, as it is too different from training
        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(validation_data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(inputs)
            loss = self.loss_function(logits, labels)
            validation_step_counter += 1
            accumulated_loss += loss.item()
            class_predictions = logits.max(dim=1)[1]
            sum_correct_predictions += torch.sum(class_predictions == labels).item()
            sum_predictions += logits.shape[0]

        avg_loss = accumulated_loss / validation_step_counter
        accuracy = sum_correct_predictions / sum_predictions
        logging.info(f"[VAL] avg loss = {avg_loss:.6f} \t acc = {accuracy:.6f}")

        mlflow.log_metric("val_loss", avg_loss, self.optimizer_step_counter)
        mlflow.log_metric("val_acc", accuracy, self.optimizer_step_counter)

        # This is an additional feature: if after validation we see we have a new best validation
        # error, we save the current state. This is built-in early stopping, as we always have the
        # best checkpoint put away for later. If one already exists, it is overwritten.
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            logging.info(f"got a new minimum loss of {self.best_loss}")
            output_path = self.output_path / "best.pth"
            logging.info(f"saving model to {output_path}")
            torch.save(self.model.state_dict(), output_path)

        # (maybe) update the learning rate
        logging.info("Updating lr scheduler.")
        self.lr_scheduler.step(avg_loss)
        # this should be "self.lr_scheduler.get_lr(), but the plateau scheduler is the only
        # scheduler without that method, so we need to access the protected member instead
        logging.info(f"Learning rate now at {self.lr_scheduler._last_lr}")
        mlflow.log_metric("lr", self.lr_scheduler._last_lr[0], self.optimizer_step_counter)

    def save_model(self):
        """Saves current state of the model to a unique subfolder"""
        output_path = self.output_path / str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S") +
                                             '_epoch{}.pth'.format(self.epoch + 1))
        logging.info(f"saving model to {output_path}")
        torch.save(self.model.state_dict(), output_path)

    def train(self):
        """Triggers the actual training - performs initialization and loops over the epochs."""
        logging.info(f"PyTorch Version {torch.__version__}")
        logging.info(f"using device {self.device}")
        logging.info(f"using architecture {self.h_params.architecture}")
        logging.info(f"using pre-trained model: {self.h_params.model_weights}")
        logging.info(f"using a batch size of {self.h_params.batch_size}")

        self.initialize_model()

        logging.info(f"using model: {self.model}")

        for _ in range(self.h_params.num_epochs):
            self.train_epoch()

        # if the model is not saved after every epoch anyway, save it now so we always have the
        # very last state.
        if self.h_params.save_every != -1:
            self.save_model()

        # log that training is over
        logging.info('Training done')


def parse_args():
    """Parses fhe current command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file (required)")
    # you may want to add a debug flag here - otherwise everything else should be in the training
    # config file
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # simple main method, we initialize the trainer...
    trainer = Trainer(parse_args())
    try:
        # ... run training
        trainer.train()
    # this special block makes sure that if a training is interrupted, we save the current state
    # this allows us to interrupt a long running training if there is some maintenance work to be
    # done or similar
    except KeyboardInterrupt:
        if trainer.model is not None:
            outputPath = trainer.output_path / \
                         str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + '_INTERRUPTED.pth')
            torch.save(trainer.model.state_dict(), outputPath)
            logging.info('Saved after interrupt')
        else:
            logging.warning("Could not save working model")
        sys.exit(0)
