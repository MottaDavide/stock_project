# How to run
Here the main steps to run properly the main notebook `nb_project.ipynb` inside the `notebook` folder, and the API inside `stock_project/API` folder.

## Install Poetry
Be sure to have [poetry](https://python-poetry.org/) installed. Here a quick guide:


### Step 1: Install Poetry
Run the following command in your terminal to install Poetry:
For **Linux/macOS**:

```
curl -sSL https://install.python-poetry.org | python3 -
```

For **Windows** (PowerShell):

```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### Step 2: Add Poetry to your PATH
Once installed, add Poetry to your environment’s PATH by following the instructions that the installer provides. Typically, you will need to add the following to your .bashrc, .zshrc, or equivalent:

```
export PATH="$HOME/.local/bin:$PATH"
```

Then, reload your shell configuration:
```
source ~/.bashrc  # or ~/.zshrc depending on your shell
```

### Step 3: Verify the installation
To verify that Poetry is installed correctly, run:

```
poetry --version
```
You should see the version of Poetry printed in the terminal.


## Install Dependencies Using Poetry

Once Poetry is installed, follow these steps to install the project’s dependencies:

### Step 1: Navigate to the project folder
Open your terminal and navigate to the root folder of your project, where the pyproject.toml file is located. Example:

```
cd path/to/your/project/stock_project
```

### Step 2: Install the dependencies
Run the following command to install the dependencies listed in the pyproject.toml file:
```
poetry install
```

This will create a virtual environment for the project and install all the required dependencies.

### Step 3: Activate the virtual environment (optional)
Poetry automatically handles virtual environments, but if you want to activate the virtual environment manually, you can use:

```
poetry shell
```
This step is optional because running commands with poetry run automatically uses the virtual environment.


## Running the FastAPI Application

This project contains a FastAPI application, you can run it as follows:

### Step 1: Run the FastAPI server
Inside the `stock_project/API` folder, run the following code (terminal)

```
poetry run uvicorn main:app --reload
```
- `main` refers to the path of the FastAPI app.
- `--reload` allows automatic reloading on code changes (useful during development).

### Step 2: Access the API
Once the server is running, you can access the FastAPI API by opening your browser and navigating to:

http://127.0.0.1:8000

You can also view the interactive API documentation provided by FastAPI at:

http://127.0.0.1:8000/docs

### Step 3: Make a Request
- Inside the `stock_project/API` folder, run the file `response.py`
- Write the wished stock code to retrieve the following information
    - stock code
    - bias (prediction over real)
    - RMSE (root mean squared error)



## Running Notebook
In `notebook` folder, open and run the `nb_project.ipynb`