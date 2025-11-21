# How to run the project

0. Make sure you have python 3.12 installed you can check that with

```sh
py -0p
```

1. Open a console in the folder `/entrega3/human-activity-an/`

2. Create a python virtual environment with

```sh
py -3.12 -m venv venv
```

3. Activate the virtual environment you just created with:


- For windows
```sh
./venv/Scripts/activate
```

- For linux
```sh
source venv/bin/activate
```

4. Install the `requirements.txt` with

```sh
pip install -r requirements.txt
```

5. To run the app, if you have them same version you used to create the venv in your path for packages use:

```sh
streamlit run app.py
```

Else, if you are using another version for packages from the one you created the venv use>

```sh
py -m streamlit run app.py
```