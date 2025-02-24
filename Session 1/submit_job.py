# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import ray
from ray import job_submission
import time

def submit_training_job():
    # Initialize Ray with dashboard
    ray.init(
        address="local",
        dashboard_port=8265,
        include_dashboard=True
    )

    # Create a job submission client
    client = job_submission.JobSubmissionClient("http://127.0.0.1:8265")

    # Submit the job
    job_id = client.submit_job(
        entrypoint="python3 ray_distributed_training.py",
        runtime_env={
            "pip": [
                "torch",
                "ray[default]"
            ]
        }
    )

    print(f"Submitted job with ID: {job_id}")
    
    # Optional: Wait for job completion and print status
    while True:
        status = client.get_job_status(job_id)
        print(f"Job status: {status}")
        if status in ["SUCCEEDED", "FAILED", "STOPPED"]:
            break
        time.sleep(10)

    # Get job logs
    logs = client.get_job_logs(job_id)
    print("Job logs:")
    print(logs)

def keep_dashboard_alive():
    try:
        while True:
            time.sleep(60)  # Sleep for a minute before checking again
    except KeyboardInterrupt:
        print("Shutting down Ray...") 
        ray.shutdown()


if __name__ == "__main__":
    submit_training_job()
    keep_dashboard_alive()