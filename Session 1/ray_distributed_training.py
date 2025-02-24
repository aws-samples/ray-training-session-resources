# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import ray
import ray.data
from ray import train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig, CheckpointConfig, FailureConfig
from collections import OrderedDict
import time

# Defining model
def create_model():
    return nn.Sequential(
        nn.Linear(764, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.Sigmoid()
    )

# Load state dict (i.e. model weights) new model object
def load_distributed_state_dict(model, state_dict):
    # If the state dict has "module." prefix, remove it
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
        
    return model.load_state_dict(new_state_dict)

def train(config):
    # Prepare model for distributed training
    model = create_model()
    model = ray.train.torch.prepare_model(model)

    # Read in config variables first
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]  

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)  

    # Import and configure dataset for distributed training
    train_dataset_shard = ray.train.get_dataset_shard("train")
    
    dataloader = train_dataset_shard.iter_torch_batches(
        batch_size=batch_size,
        dtypes=torch.float
    )

    # train epoch across dataloader batches
    for epoch in range(num_epochs):
        batch_index = 0
        running_loss = 0.0

        for batch in dataloader:
            label = batch["target"].long()
            del batch["target"]
            
            features = []
            for key in sorted(batch.keys()):
                features.append(batch[key].unsqueeze(1))
            
            inputs = torch.cat(features, dim=1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            metrics = {
                "epoch": epoch,
                "batch": batch_index,
                "loss": running_loss / (batch_index + 1)
            }

            # Checkpoint once per epoch
            if batch_index == 100:
                with tempfile.TemporaryDirectory() as temp_dir:
                    checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
                    
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "batch": batch_index,
                        "loss": running_loss / (batch_index + 1)
                    }, checkpoint_path)
                    
                    checkpoint = ray.train.Checkpoint.from_directory(temp_dir)
                    
                    print(f"Saving checkpoint at epoch {epoch}, batch {batch_index}")
                    ray.train.report(metrics, checkpoint=checkpoint)

            running_loss = 0.0
            batch_index += 1

def main(config=None):
    # Initialize Ray cluster connection (connects to existing cluster)
    # ray.init(
    #     address="local",
    #     dashboard_port=8265,  # Default Ray dashboard port
    #     include_dashboard=True
    # )

    if config is None:
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 10
        }

    # # Define training config
    # config = {
    #     "learning_rate":  0.001,
    #     "batch_size": 32,
    #     "num_epochs": 10
    # }

    # Determine Ray resource allocation (vary depending on GPU presence)
    if torch.cuda.is_available():
        resources_per_worker = {
            "CPU": 1,  
            "memory": 1 * 1024 * 1024 * 1024,  
            "GPU": 1
        }
    else:
        resources_per_worker = {
            "CPU": 1,  
            "memory": 1 * 1024 * 1024 * 1024 
        }

    # Define scaling for distributed training
    scaling_config = ScalingConfig(
        num_workers=6,
        use_gpu=torch.cuda.is_available(),
        resources_per_worker=resources_per_worker
    )

    # Configure checkpointing and error logging
    checkpoint_config = CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="loss",
        checkpoint_score_order="min",
    )

    run_config = RunConfig(
        checkpoint_config = checkpoint_config,
        failure_config = FailureConfig(max_failures=3),
    )

    # Loading dataset from local
    dataset = ray.data.read_csv(
        "sample_data.csv",
        ray_remote_args={
            "num_cpus": 12,
            "memory": 12 * 1024 * 1024 * 1024
        }
    )

    # Split into train/test
    train_dataset, test_dataset = dataset.train_test_split(test_size=0.2)

    datasets = {"train": train_dataset, "test": test_dataset}

    # Create a trainer
    trainer = TorchTrainer(
        train_loop_per_worker = train,
        train_loop_config = config,
        scaling_config = scaling_config,
        run_config = run_config,
        datasets = datasets
    )

    try:
        # Start distributed training
        print("Starting distributed training...")
        start_train = time.time()
        result = trainer.fit()
        
        # Check if we have a valid result and checkpoint (NOTE - this executes only after the entire training process is finished to find the best model checkpoint, save it for later inference, and print the final output stats)
        if result and result.checkpoint:
            print("Training completed successfully!")
            
            # Get the best checkpoint
            best_checkpoint = result.checkpoint
            
            # Create a temporary directory to download the checkpoint
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the checkpoint from S3 to local temporary directory
                downloaded_checkpoint = best_checkpoint.to_directory(temp_dir)
                
                # Load the checkpoint file
                checkpoint_data = torch.load(
                    os.path.join(downloaded_checkpoint, "checkpoint.pt"),
                    weights_only=True  # Only load last checkpoint model weights
                )
                
                # Create new model and load current training state
                model = create_model()
                load_distributed_state_dict(model, checkpoint_data["model_state_dict"])
                
                # Save the model locally
                torch.save(model.state_dict(), "./distributed_model.pth")
                print("Model saved successfully!")
                
                # Print final metrics
                print(f"Final training metrics:")
                print(f"Epochs: {checkpoint_data['epoch']}")
                print(f"Loss: {checkpoint_data['loss']:.4f}")
        
        else:
            print("Training completed but no checkpoint was saved.")

        print(f"Total training time: {time.time() - start_train} seconds")
    
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

    # Shut down ray
    ray.shutdown()

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total run time: {time.time() - start} seconds")