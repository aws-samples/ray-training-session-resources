# How to use code sample:
1. Git clone this repo
2. `cd Session\ 1/`
3. `python3 -m venv session_1_venv`
4. `source session_1_venv/bin/activate`
5. `pip install -r requirements.txt`
6. `python3 generate_dataset.py`
7. `python3 simple_training # run simple local model`
8. `python3 sumbit_job.py # run ray training equivalent`

NOTE - the ray code will continue running until you explicitly cancel it, as it is designed to make it easy to view the Ray Dashboard even after the training run is finished.